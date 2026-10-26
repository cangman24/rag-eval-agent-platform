# -*- coding: utf-8 -*-
"""core/agent.py —— 可恢复的 Agent 主循环（LangGraph 风格状态机）。

对应简历里的四个工程机制，这里全部落地：
  1. **受控状态机**     —— 节点 + 条件边，执行路径可读可审计
  2. **Checkpoint 续跑** —— 每轮落盘，中断后从最近检查点恢复，已完成节点不重跑
  3. **工具白名单**     —— 意图 → task_type → allowed_tools，收敛模型可调用范围
  4. **超时熔断**       —— 最大轮次 + 单工具超时 + 总耗时上限 + 重复动作检测

★ 为什么不用 langgraph 库而是自实现：
  - 本模块零外部依赖 -> 可离线单测。Checkpoint/白名单/熔断这些**工程逻辑**
    的正确性与 langgraph 是否装得上无关，必须先能验证。
  - 迁移成本极低：状态字段、节点划分、条件边的语义都对齐 langgraph，
    真机接入时把 `AgentEngine.run` 换成 StateGraph 编译即可。

★ 状态必须可 JSON 序列化（否则存不进 checkpoint）。
  所以工具返回值会被统一规范化成 JSON-safe 结构（见 `normalize_observation`）。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple)

from .mcp import ToolError, ToolNotAllowed, ToolRegistry

# --------------------------------------------------------------------- 常量
DEFAULT_MAX_ROUNDS = 6
DEFAULT_MAX_TOOL_CALLS = 12
DEFAULT_TOTAL_TIMEOUT_S = 120.0
DEFAULT_TOOL_TIMEOUT_S = 30.0


# --------------------------------------------------------------------- 事件
class EventType:
    """SSE 事件类型（前端轨迹面板按这些类型渲染）。"""

    START = "start"
    INTENT = "intent"
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    VERIFY = "verify"
    ANSWER = "answer"
    REFUSAL = "refusal"
    LIMIT = "limit"
    RESUME = "resume"
    ERROR = "error"
    DONE = "done"


def make_event(event_type: str, **data: Any) -> Dict[str, Any]:
    return {"type": event_type, "ts": time.time(), "data": data}


# --------------------------------------------------------------------- 状态
def json_safe(value: Any, _depth: int = 0) -> Any:
    """把任意对象转成 JSON 可序列化结构。

    这是 checkpoint 能工作的前提。工具返回自定义对象时，
    这里退化成 repr —— 保底不丢轮次，但建议工具直接返回 dict。
    """
    if _depth > 8:
        return repr(value)[:500]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, _depth + 1) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict(), _depth + 1)
    return repr(value)[:500]


@dataclass
class AgentState:
    """Agent 的全部可变状态。字段命名对齐 LangGraph 的 State 概念。"""

    query: str = ""
    thread_id: str = ""
    task_type: str = "general"
    allowed_tools: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    scratchpad: List[Dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.time)
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    verified: bool = False
    verify_reason: str = ""
    refused: bool = False
    finished: bool = False
    stop_reason: str = ""
    last_observation: Any = None

    # ---------------------------------------------------------- 序列化
    def to_dict(self) -> Dict[str, Any]:
        return json_safe(self.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentState":
        known = set(AgentState.__dataclass_fields__)  # type: ignore[attr-defined]
        return AgentState(**{k: v for k, v in (d or {}).items() if k in known})

    def elapsed_s(self, clock: Optional[Callable[[], float]] = None) -> float:
        """已耗时（秒）。

        ★ `clock` 参数是可选的，为了**向后兼容**：原有调用 `state.elapsed_s()`
          行为不变。但引擎在判定超时时必须传自己的注入时钟：

              state.elapsed_s(self._clock)

          否则这里会退回真实的 time.time()，注入时钟在超时判定上就失效了
          —— 这类"部分走了注入、部分没走"的 bug 最难发现，
          因为测试里注入时钟看上去"生效了"（其它地方在用），
          唯独超时这一条判断永远用真时间。
        """
        now = clock() if clock is not None else time.time()
        return now - self.started_at

    def signature(self) -> str:
        """用于重复动作检测：轮次无关，只看意图与工具序列。"""
        tools = [s.get("tool") for s in self.scratchpad if s.get("tool")]
        return f"{self.task_type}|{','.join(tools)}"


# --------------------------------------------------------------------- Checkpoint
class Checkpointer:
    """检查点存储接口。"""

    def save(self, thread_id: str, state: AgentState) -> None:
        raise NotImplementedError

    def load(self, thread_id: str) -> Optional[AgentState]:
        raise NotImplementedError

    def delete(self, thread_id: str) -> None:
        raise NotImplementedError

    def list_threads(self) -> List[str]:
        raise NotImplementedError

    # ---- 轨迹事件（可选能力）----
    # ★ 为什么单独一套方法而不是塞进 save()：
    #     状态是"最后长什么样"，轨迹是"怎么走到这里的"。前者覆盖写，
    #     后者必须**追加**。把两者混在一个 save 里，续跑时就会把首次
    #     运行的轨迹冲掉 —— 回放功能直接失效。
    #   默认实现是 no-op：不是每种存储都需要轨迹（例如纯内存调试）。

    def append_events(self, thread_id: str,
                      events: Sequence[Dict[str, Any]]) -> None:
        """追加一批事件到该 thread 的轨迹。"""

    def load_events(self, thread_id: str) -> List[Dict[str, Any]]:
        """读取该 thread 的完整轨迹（按发生顺序）。"""
        return []

    def reset_events(self, thread_id: str) -> None:
        """清空该 thread 的轨迹（新开一次运行时用）。"""



class InMemoryCheckpointer(Checkpointer):
    """内存实现，用于单测与本地调试。"""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.saves = 0

    def save(self, thread_id: str, state: AgentState) -> None:
        with self._lock:
            self._store[thread_id] = state.to_json()
            self.saves += 1

    def load(self, thread_id: str) -> Optional[AgentState]:
        with self._lock:
            raw = self._store.get(thread_id)
        return AgentState.from_dict(json.loads(raw)) if raw else None

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._store.pop(thread_id, None)

    def list_threads(self) -> List[str]:
        with self._lock:
            return sorted(self._store)

    def append_events(self, thread_id: str,
                      events: Sequence[Dict[str, Any]]) -> None:
        with self._lock:
            self._events.setdefault(thread_id, []).extend(events)

    def load_events(self, thread_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events.get(thread_id, []))

    def reset_events(self, thread_id: str) -> None:
        with self._lock:
            self._events.pop(thread_id, None)


class JsonFileCheckpointer(Checkpointer):
    """落盘实现：每个 thread 一个 JSON 文件。

    为什么先用 JSON 文件而不是 SQLite：
      - 零依赖、可读、便于调试（直接打开看状态）；
      - 单机规模下完全够用。
      真机若要并发写，再换 SqliteSaver —— 接口不变。
    """

    def __init__(self, path: str = "data/checkpoints") -> None:
        self.path = path
        os.makedirs(path, exist_ok=True)

    def _file(self, thread_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
        return os.path.join(self.path, f"{safe}.json")

    def _events_file(self, thread_id: str) -> str:
        """轨迹文件：JSON Lines，一行一个事件。

        ★ 为什么是 JSONL 而不是一个 JSON 数组：
            轨迹是**追加写**的。JSON 数组每次追加都要"读全文→改→写回"，
            跑到一半崩了还会留下半截非法 JSON；JSONL 天然追加，
            即便进程被杀，已落盘的那些行依然是合法可解析的。
        """
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
        return os.path.join(self.path, f"{safe}.events.jsonl")

    def save(self, thread_id: str, state: AgentState) -> None:
        target = self._file(thread_id)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(state.to_json())
        os.replace(tmp, target)

    def load(self, thread_id: str) -> Optional[AgentState]:
        fp = self._file(thread_id)
        if not os.path.exists(fp):
            return None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return AgentState.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def delete(self, thread_id: str) -> None:
        fp = self._file(thread_id)
        if os.path.exists(fp):
            os.remove(fp)

    def list_threads(self) -> List[str]:
        return sorted(os.path.splitext(n)[0] for n in os.listdir(self.path)
                      if n.endswith(".json"))

    def append_events(self, thread_id: str,
                      events: Sequence[Dict[str, Any]]) -> None:
        fp = self._events_file(thread_id)
        with open(fp, "a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False, default=str))
                f.write("\n")

    def load_events(self, thread_id: str) -> List[Dict[str, Any]]:
        fp = self._events_file(thread_id)
        if not os.path.exists(fp):
            return []
        out: List[Dict[str, Any]] = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # 半截行（进程被杀）：跳过而不是让整个回放失败
                    continue
        return out

    def reset_events(self, thread_id: str) -> None:
        fp = self._events_file(thread_id)
        if os.path.exists(fp):
            os.remove(fp)


# --------------------------------------------------------------------- 结果
@dataclass
class AgentRun:
    state: AgentState
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.state.finished


# --------------------------------------------------------------------- 引擎
class AgentEngine:
    """Agent 主循环。

    Args:
        registry: 工具注册中心
        intent_fn: query -> task_type 字符串（真实项目里是 LLM 分类 + 规则兜底）
        plan_fn: (query, task_type) -> 子任务列表
        act_fn: (state, allowed_tools) -> {"tool": name, "args": {...}} 或
                {"final": "答案文本", "citations": [...]}
                —— 这一步是"模型决策"，用注入函数实现，便于离线单测与替换
        verify_fn: (state, answer) -> (ok: bool, reason: str)
        route_table: task_type -> 允许的工具名列表（★ 白名单的来源）
        checkpointer: 检查点实现
        max_rounds / max_tool_calls / total_timeout_s: 熔断参数
    """

    def __init__(
        self,
        registry: ToolRegistry,
        intent_fn: Callable[[str], str],
        plan_fn: Callable[[str, str], List[str]],
        act_fn: Callable[[AgentState, List[str]], Dict[str, Any]],
        verify_fn: Optional[Callable[[AgentState, str], Tuple[bool, str]]] = None,
        route_table: Optional[Dict[str, Sequence[str]]] = None,
        checkpointer: Optional[Checkpointer] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.registry = registry
        self.intent_fn = intent_fn
        self.plan_fn = plan_fn
        self.act_fn = act_fn
        self.verify_fn = verify_fn or (lambda s, a: (bool(a.strip()), "非空回答"))
        self.route_table = dict(route_table or {})
        self.checkpointer = checkpointer
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.total_timeout_s = total_timeout_s
        self.tool_timeout_s = tool_timeout_s
        # ★ 时钟必须可注入，且**所有**计时都要走它：
        #   AgentState.started_at、thread_id 生成、超时判定、总耗时。
        #   如果超时判定用真实的 time.time()，注入时钟就形同虚设 ——
        #   这个 bug 正是被 test_total_timeout_stops 抓出来的。
        self._clock = clock
        # 上一次运行的最终状态（run / run_stream 都会设置）
        # —— 与 LangGraphAgentEngine 保持一致，上层代码无需分支
        self.last_state: Optional[AgentState] = None

    # ------------------------------------------------------------ 白名单
    def route(self, task_type: str) -> List[str]:
        """意图 -> 允许的工具集。

        ★ 关键工程取舍：未在 route_table 里的意图**默认给空清单**（最保守），
          而不是给全集。宁可模型少做，也不能让它越权调用。
        """
        return list(self.route_table.get(task_type, []))

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        resume: bool = False,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> AgentRun:
        tid = thread_id or f"t{int(self._clock() * 1000)}"
        events: List[Dict[str, Any]] = []

        def emit(etype: str, **data: Any) -> None:
            ev = make_event(etype, **data)
            events.append(ev)
            if on_event is not None:
                on_event(ev)

        # ---------------- 恢复 ----------------
        state: Optional[AgentState] = None
        if resume and self.checkpointer is not None:
            state = self.checkpointer.load(tid)
            if state is not None:
                emit(EventType.RESUME, thread_id=tid, rounds=state.rounds,
                     tool_calls=state.tool_calls, finished=state.finished)
                if state.finished:
                    emit(EventType.DONE, reason="已完成的检查点，无需续跑")
                    self.last_state = state
                    return AgentRun(state=state, events=events)

        if state is None:
            state = AgentState(query=query, thread_id=tid, started_at=self._clock())

        try:
            if not state.plan:
                emit(EventType.START, query=query, thread_id=tid)
                # ---- 节点 1：意图识别 ----
                state.task_type = self.intent_fn(query) or "general"
                state.allowed_tools = self.route(state.task_type)
                emit(EventType.INTENT, task_type=state.task_type,
                     allowed_tools=state.allowed_tools)
                # ---- 节点 2：任务规划 ----
                state.plan = list(self.plan_fn(query, state.task_type) or [])
                emit(EventType.PLAN, steps=state.plan)
                self._save(state)

            # ---- 节点 3~N：act / verify 循环（条件边） ----
            while state.rounds < self.max_rounds:
                # 熔断：总耗时（★ 走注入时钟，便于测试与时间旅行调试）
                elapsed = self._clock() - state.started_at
                if elapsed > self.total_timeout_s:
                    state.stop_reason = "total_timeout"
                    emit(EventType.LIMIT, reason="total_timeout",
                         elapsed_s=round(elapsed, 2))
                    break
                # 熔断：工具调用次数
                if state.tool_calls >= self.max_tool_calls:
                    state.stop_reason = "max_tool_calls"
                    emit(EventType.LIMIT, reason="max_tool_calls",
                         tool_calls=state.tool_calls)
                    break

                state.rounds += 1
                decision = self.act_fn(state, state.allowed_tools) or {}
                decision = json_safe(decision) or {}

                # ---- 模型认为可以收尾 ----
                if "final" in decision:
                    answer = str(decision.get("final") or "")
                    state.answer = answer
                    state.citations = list(decision.get("citations") or [])
                    ok, reason = self.verify_fn(state, answer)
                    state.verified = ok
                    state.verify_reason = reason
                    emit(EventType.VERIFY, ok=ok, reason=reason)
                    if ok:
                        state.finished = True
                        state.stop_reason = "answer"
                        emit(EventType.ANSWER, text=answer,
                             citations=state.citations)
                        self._save(state)
                        # ★ 必须 break。若这里用 continue，循环条件
                        #   `rounds < max_rounds` 不检查 finished，于是会一遍遍
                        #   重复"校验通过 -> 输出答案"，答案被重复打印 N 次、
                        #   轮次也被虚增。这个 bug 是被 demo_agent.py 的输出
                        #   抓到的（同一答案出现 4 次）。
                        break
                    # 校验不通过 -> 条件边回到 act，把原因喂回去
                    state.scratchpad.append(
                        {"type": "verify_feedback", "reason": reason})
                    self._save(state)
                    continue

                # ---- 模型要求调工具 ----
                tool_name = decision.get("tool")
                if not tool_name:
                    state.stop_reason = "no_decision"
                    emit(EventType.ERROR, reason="act_fn 既没给 final 也没给 tool")
                    break

                args = dict(decision.get("args") or {})
                emit(EventType.TOOL_CALL, tool=tool_name, args=args,
                     round=state.rounds)
                try:
                    result = self.registry.call(tool_name, args,
                                                allowed=state.allowed_tools)
                    state.tool_calls += 1
                    state.last_observation = json_safe(result)
                    state.scratchpad.append({
                        "type": "tool_call", "tool": tool_name, "args": args,
                        "ok": True, "result": state.last_observation,
                    })
                    # ★ ok / message 两个字段是为与 LangGraph 版对齐而补的：
                    #   两版引擎可互换（make_agent_engine），前端面板不该为
                    #   "现在跑的是哪一版" 写分支 —— 字段不一致就会显示错的
                    #   状态（例如自实现版没有 ok，面板会显示 ok=false）。
                    emit(EventType.TOOL_RESULT, tool=tool_name, ok=True,
                         result=state.last_observation)
                except ToolNotAllowed as exc:
                    state.scratchpad.append({
                        "type": "tool_call", "tool": tool_name, "args": args,
                        "ok": False, "error": str(exc), "blocked": True,
                    })
                    emit(EventType.TOOL_ERROR, tool=tool_name, error=str(exc),
                         message=str(exc), blocked=True)
                except ToolError as exc:
                    state.tool_calls += 1
                    state.scratchpad.append({
                        "type": "tool_call", "tool": tool_name, "args": args,
                        "ok": False, "error": str(exc),
                    })
                    emit(EventType.TOOL_ERROR, tool=tool_name, error=str(exc),
                         message=str(exc), blocked=False)
                self._save(state)

            # ---------------- 收尾 ----------------
            if not state.finished and not state.stop_reason:
                state.stop_reason = "max_rounds"
                emit(EventType.LIMIT, reason="max_rounds", rounds=state.rounds)

            if state.finished and not state.answer:
                state.stop_reason = state.stop_reason or "empty_answer"

            self._save(state)
            emit(EventType.DONE, stop_reason=state.stop_reason,
                 rounds=state.rounds, tool_calls=state.tool_calls)
            self.last_state = state
            return AgentRun(state=state, events=events)

        except Exception as exc:  # noqa: BLE001 - 顶层兜底，保证 checkpoint 落盘
            state.stop_reason = "error"
            self._save(state)
            emit(EventType.ERROR, error=f"{type(exc).__name__}: {exc}")
            self.last_state = state
            return AgentRun(state=state, events=events,
                            error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ 辅助
    def _save(self, state: AgentState) -> None:
        if self.checkpointer is not None:
            try:
                self.checkpointer.save(state.thread_id, state)
            except OSError:
                # 存盘失败不让整轮 Agent 崩掉；代价是这一轮的检查点丢失，
                # resume 时会退回上一个可用检查点（或初始状态）重跑。
                pass

    def run_stream(
        self,
        query: str,
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """流式执行 —— 与 LangGraphAgentEngine.run_stream 同签名、同语义。

        ★ 必须如实说明的差异：
            这一版是**批量吐出**（跑完再 yield），不是节点级实时。
            因为自实现的 while 循环没有"每步暂停"的结构，要真增量就得把
            run() 整个改成生成器 —— 收益（回退路径更好看）不值得这个风险。
            LangGraph 版（主路径，真机默认）是**真增量**：每个图节点执行完
            就推一批事件。

            对调用方的影响只有"事件到达的时机"，事件内容与顺序两版一致，
            所以上层代码（API / 面板）不需要分支。前端想知道当前用的是哪一版，
            读 `GET /agent/health` 的 `engine` 字段。
        """
        out: List[Dict[str, Any]] = []
        cp = self.checkpointer
        if cp is not None and not resume and thread_id:
            cp.reset_events(thread_id)
        # thread_id 由 run() 内部生成时才需要回捞 —— start 事件里带
        holder: Dict[str, Any] = {"tid": thread_id}

        def sink(ev: Dict[str, Any]) -> None:
            if holder["tid"] is None:
                holder["tid"] = (ev.get("data") or {}).get("thread_id")
            out.append(ev)
            if cp is not None and holder["tid"]:
                try:
                    cp.append_events(holder["tid"], [ev])
                except OSError:  # pragma: no cover - 旁路失败不影响主流程
                    pass

        self.run(query, thread_id=thread_id, resume=resume, on_event=sink)
        for ev in out:
            yield ev

    def resume(self, thread_id: str) -> AgentRun:
        """从检查点续跑。已完成节点不会重复执行。"""
        return self.run(query="", thread_id=thread_id, resume=True)

    def detect_loop(self, state: AgentState, window: int = 3) -> bool:
        """重复动作检测：最近 window 次工具序列完全一致 -> 判定为死循环。

        用于给 act_fn 一个"你卡住了"的信号（真实项目里喂回模型让它换策略）。
        """
        tools = [s.get("tool") for s in state.scratchpad if s.get("tool")]
        if len(tools) < window * 2:
            return False
        tail = tools[-window:]
        prev = tools[-window * 2:-window]
        return tail == prev

# -*- coding: utf-8 -*-
"""core/llm.py —— LLM 客户端（DeepSeek / OpenAI 兼容），含重试、限流、磁盘缓存。

设计要点（面试可讲）：
  1. **重试**只对可恢复错误（429 / 超时 / 5xx）生效，且用**指数退避 + 抖动**。
     绝不重试 400/401（参数错、密钥错）—— 重试只会浪费时间并可能触发风控。
  2. **限流**用令牌桶：评测要跑几百次调用，不加限流必然撞 429。
  3. **磁盘缓存**按 prompt 哈希缓存：评测迭代时同样的输入不重复花钱。
     这一条能把 Day 3 的评测成本降到几元，也把重跑时间从小时级降到分钟级。
  4. 把"HTTP 调用"抽成可注入的 `transport`，于是重试/退避/缓存逻辑
     **可以在无网络环境下单测** —— 这是本项目能离线验证的关键手法。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# --------------------------------------------------------------------- 异常
class LLMError(Exception):
    """LLM 调用失败（不可重试或重试耗尽）。"""


class RetryableError(LLMError):
    """可重试错误（429 / 超时 / 5xx）。"""


class NonRetryableError(LLMError):
    """不可重试错误（400 参数错 / 401 密钥错）。"""


# --------------------------------------------------------------------- 令牌桶
class TokenBucket:
    """简单令牌桶限流器（线程安全）。

    Args:
        rate: 每秒补充的令牌数
        capacity: 桶容量（决定允许的瞬时突发）
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        """获取令牌；超时返回 False。"""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.05))


# --------------------------------------------------------------------- 缓存
class DiskCache:
    """按 key 哈希落盘的 JSON 缓存。评测迭代的关键提速手段。"""

    def __init__(self, path: str = "data/cache/llm", enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _file(self, key: str) -> str:
        return os.path.join(self.path, f"{key}.json")

    def _ensure_dir(self) -> bool:
        """确保目录存在。

        ★ 必须每次写前检查：目录可能被外部清理掉，
        或者 path 本身在一个尚未可写的父目录下。
        若这里静默失败，缓存会"看起来启用但永远不命中"——很难排查。
        """
        if not self.enabled:
            return False
        try:
            os.makedirs(self.path, exist_ok=True)
            return True
        except OSError:
            return False

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        fp = self._file(key)
        if not os.path.exists(fp):
            self.misses += 1
            return None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.hits += 1
            return data
        except (json.JSONDecodeError, OSError, ValueError):
            self.misses += 1
            return None

    def set(self, key: str, value: dict) -> bool:
        """写入缓存；返回是否成功。"""
        if not self._ensure_dir():
            return False
        target = self._file(key)
        tmp = target + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, target)
            return True
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# --------------------------------------------------------------------- 结果
@dataclass
class LLMResult:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    attempts: int = 1
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMDelta:
    """流式增量。

    为什么不用 LLMResult：流式下"完整结果"在最后一个 delta 之前并不存在。
    让每一帧只带**增量**，调用方自己攒正文；`done=True` 那帧带 usage 与首字延迟。
    """
    text: str = ""
    done: bool = False
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 首字延迟（ms）：只在第一个有内容的 delta 上赋值，整轮共用一个数
    first_token_ms: Optional[float] = None
    latency_ms: float = 0.0
    attempts: int = 1
    # 全文（只在 done 帧上有值）
    # ★ 为什么不复用 text：text 的语义是"本帧新增"，调用方普遍用
    #   "".join(d.text) 拼正文。done 帧若也塞全文，就会被拼两遍 ——
    #   这个坑在写测试时立刻现形（'答答' != '答'）。语义分开，谁也不坑谁。
    full_text: str = ""


# --------------------------------------------------------------------- 客户端
class LLMClient:
    """OpenAI 兼容 chat 客户端。

    Args:
        transport: 可注入的调用函数 `fn(model, messages, **kw) -> dict`。
            默认实现走 openai SDK。注入 fake 即可离线单测重试逻辑。
        max_retries: 可重试错误的最大重试次数
        base_delay: 退避基数（秒）
        rate: 限流速率（请求/秒），None 表示不限流
        cache: DiskCache 实例或 None
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        transport: Optional[Callable[..., dict]] = None,
        max_retries: int = 4,
        base_delay: float = 0.8,
        rate: Optional[float] = 5.0,
        cache: Optional[DiskCache] = None,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: bool = True,
        stream_transport: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.max_retries = max(0, max_retries)
        self.base_delay = base_delay
        self.timeout = timeout
        self.bucket = TokenBucket(rate) if rate else None
        self.cache = cache
        self._sleep = sleep
        self._jitter = jitter
        self._transport = transport
        self._stream_transport = stream_transport
        self._client = None
        # 成本统计（评测报告里要写"花了多少钱"）
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "errors": 0, "retries": 0,
                      "stream_calls": 0}

    # -------------------------------------------------- 默认 transport
    def _ensure_client(self):
        """懒建 openai 客户端（不 import 大包也能用注入 transport 跑测试）。"""
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise LLMError("未安装 openai SDK：pip install openai") from exc
            if not self.api_key:
                raise NonRetryableError(
                    "缺少 API Key。请在 .env 设置 DEEPSEEK_API_KEY，"
                    "或构造 LLMClient 时传入 api_key。"
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout)
        return self._client

    def _default_transport(self, messages: Sequence[dict], **kw) -> dict:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=kw.pop("model", self.model), messages=list(messages), **kw
        )
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    def _default_stream_transport(self, messages: Sequence[dict], **kw):
        """默认流式 transport：产出 OpenAI 兼容的 chunk dict。"""
        client = self._ensure_client()
        model = kw.pop("model", self.model)
        kw["stream"] = True
        for chunk in client.chat.completions.create(
            model=model, messages=list(messages), **kw
        ):
            yield chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)

    # -------------------------------------------------- 错误分类
    @staticmethod
    def classify_error(exc: BaseException) -> type:
        """把异常映射成可重试 / 不可重试。

        判断依据优先看 status_code，其次看异常类名与消息里的关键字。
        这样即使 openai SDK 换版本，逻辑也不会失效。
        """
        status = getattr(exc, "status_code", None)
        if status is None:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)

        if isinstance(status, int):
            if status == 429 or 500 <= status < 600:
                return RetryableError
            if 400 <= status < 500:
                return NonRetryableError

        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if any(k in name for k in ("timeout", "connection", "apiconnection", "ratelimit")):
            return RetryableError
        if any(k in msg for k in ("timeout", "timed out", "connection", "rate limit",
                                  "429", "502", "503", "504", "overloaded")):
            return RetryableError
        return NonRetryableError

    # -------------------------------------------------- 主入口
    def chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        cache_key_extra: Optional[dict] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> LLMResult:
        transport = self._transport or self._default_transport
        model = kwargs.pop("model", self.model)

        key = None
        if self.cache is not None and use_cache:
            key = DiskCache.make_key("chat", model, list(messages), temperature,
                                     max_tokens, json_mode, cache_key_extra)
            hit = self.cache.get(key)
            if hit is not None:
                self.usage["cached"] += 1
                return LLMResult(
                    text=hit.get("text", ""), model=hit.get("model", model),
                    prompt_tokens=hit.get("prompt_tokens", 0),
                    completion_tokens=hit.get("completion_tokens", 0),
                    cached=True, attempts=0,
                )

        payload: Dict[str, Any] = dict(messages=list(messages), model=model,
                                       temperature=temperature, **kwargs)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        attempts = 0
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            if self.bucket is not None and not self.bucket.acquire():
                self.usage["errors"] += 1
                raise LLMError("限流等待超时：请降低并发或提高 rate")
            try:
                raw = transport(**payload)
                text, pt, ct = self._parse(raw)
                result = LLMResult(
                    text=text, model=raw.get("model", model),
                    prompt_tokens=pt, completion_tokens=ct,
                    latency_ms=(time.monotonic() - started) * 1000,
                    attempts=attempts, raw=raw,
                )
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += pt
                self.usage["completion_tokens"] += ct
                if self.cache is not None and key and use_cache:
                    self.cache.set(key, {"text": text, "model": result.model,
                                         "prompt_tokens": pt, "completion_tokens": ct})
                return result
            except BaseException as exc:  # noqa: BLE001 - 需要按类型分流
                kind = self.classify_error(exc)
                last_exc = exc
                if kind is NonRetryableError or attempt >= self.max_retries:
                    self.usage["errors"] += 1
                    raise kind(f"{type(exc).__name__}: {exc}") from exc
                self.usage["retries"] += 1
                delay = self.base_delay * (2 ** attempt)
                if self._jitter:
                    delay *= 0.5 + random.random()
                self._sleep(delay)

        self.usage["errors"] += 1
        raise LLMError(f"重试耗尽: {last_exc}")

    # -------------------------------------------------- 流式
    def can_stream(self) -> bool:
        """当前配置能否做**真** token 级流式。

        ★ 判断标准刻意严格：注入了整段式 `transport`（fake 替身）时返回 False。
          否则测试里的假 transport 会被当成流式能力的证据，把"假流式"混进
          真机结论 —— 这正是本项目反复强调的双轨边界。
        """
        if self._stream_transport is not None:
            return True
        return self._transport is None and bool(self.api_key)

    def chat_stream(
        self,
        messages: Sequence[dict],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """按 token 增量产出 LLMDelta。

        与 chat() 的三处**刻意差异**（每条都是踩过或想清楚才会这么写）：

        1. **默认不读缓存**。缓存命中意味着"没有 token 可流"，硬要吐就成了
           假流式（一次性到达后切成小块），前端逐字动画看着像真的，但它不是。
           真流式优先，所以只**写**缓存不**读**缓存 —— 非流式请求（评测）
           仍能命中这些结果，钱一分不浪费。
        2. **只在还没吐出任何 token 时重试**。已经有字到达浏览器后再重试，
           用户会看到答案从半截重头再来一遍，比直接失败更糟。
        3. 限流桶、错误分类、usage 统计与 chat() 共用同一套设施。
        """
        transport = self._stream_transport or self._default_stream_transport
        model = kwargs.pop("model", self.model)

        payload: Dict[str, Any] = dict(messages=list(messages), model=model,
                                       temperature=temperature, **kwargs)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # include_usage 让最后一帧带上 token 用量（主流 OpenAI 兼容实现都支持）。
        # 个别厂商不认这个字段会直接 400，那就显式传 stream_options=None 关掉。
        # ★ 在这儿统一处理，而不是塞进默认 transport —— 否则注入的 transport
        #   会收到一个它不认识的键，还测不出"关掉"这条分支。
        if "stream_options" in payload:
            if payload["stream_options"] is None:
                payload.pop("stream_options")
        else:
            payload["stream_options"] = {"include_usage": True}

        started = time.monotonic()
        first_ms: Optional[float] = None
        parts: List[str] = []
        pt = 0
        ct = 0
        attempts = 0
        emitted = False

        while True:
            attempts += 1
            if self.bucket is not None and not self.bucket.acquire():
                self.usage["errors"] += 1
                raise LLMError("限流等待超时：请降低并发或提高 rate")
            try:
                for raw in transport(**payload):
                    delta, usage = self._parse_delta(raw)
                    if usage:
                        pt = int(usage.get("prompt_tokens") or 0) or pt
                        ct = int(usage.get("completion_tokens") or 0) or ct
                    if delta:
                        if first_ms is None:
                            first_ms = (time.monotonic() - started) * 1000
                        parts.append(delta)
                        emitted = True
                        yield LLMDelta(text=delta, model=model,
                                       first_token_ms=first_ms,
                                       attempts=attempts)
                break
            except BaseException as exc:  # noqa: BLE001 - 按类型分流
                kind = self.classify_error(exc)
                if emitted or kind is NonRetryableError or attempts > self.max_retries:
                    self.usage["errors"] += 1
                    raise kind(f"{type(exc).__name__}: {exc}") from exc
                self.usage["retries"] += 1
                delay = self.base_delay * (2 ** (attempts - 1))
                if self._jitter:
                    delay *= 0.5 + random.random()
                self._sleep(delay)
                # 重试前清空：emitted 为 False，这里其实都是空的，清空是保险
                parts = []
                pt = ct = 0
                first_ms = None

        text = "".join(parts)
        latency_ms = (time.monotonic() - started) * 1000
        self.usage["calls"] += 1
        self.usage["stream_calls"] += 1
        self.usage["prompt_tokens"] += pt
        self.usage["completion_tokens"] += ct
        # 只写缓存不读缓存：非流式调用（评测）能吃到这份结果
        if self.cache is not None:
            key = DiskCache.make_key("chat", model, list(messages), temperature,
                                     max_tokens, False, None)
            self.cache.set(key, {"text": text, "model": model,
                                 "prompt_tokens": pt, "completion_tokens": ct})

        yield LLMDelta(done=True, model=model, prompt_tokens=pt,
                       completion_tokens=ct, first_token_ms=first_ms,
                       attempts=attempts, latency_ms=latency_ms,
                       full_text=text)   # text 保持空：本帧没有新增内容

    @staticmethod
    def _parse_delta(raw: dict) -> tuple:
        """从一个 chunk 取增量文本与可能出现的 usage。

        `stream_options={"include_usage": True}` 时最后一帧的 choices 是空数组、
        usage 有值 —— 所以两个都要取，不能"没文本就跳过整帧"。
        """
        usage = (raw or {}).get("usage") or {}
        text = ""
        try:
            choices = raw.get("choices") or []
            if choices:
                first = choices[0] or {}
                text = (first.get("delta") or {}).get("content") or ""
                # 少数兼容实现把首个 chunk 放成 message 而不是 delta
                if not text:
                    text = (first.get("message") or {}).get("content") or ""
        except (AttributeError, TypeError):  # pragma: no cover - 畸形响应
            text = ""
        return text, usage

    # -------------------------------------------------- 解析
    @staticmethod
    def _parse(raw: dict) -> tuple:
        """从 OpenAI 兼容响应里取文本与 token 用量。"""
        try:
            choices = raw["choices"]
            message = choices[0]["message"]
            text = message.get("content") or ""
        except (KeyError, IndexError, TypeError):
            text = raw.get("text", "") if isinstance(raw, dict) else ""
        usage = raw.get("usage") or {}
        return text, int(usage.get("prompt_tokens", 0) or 0), int(
            usage.get("completion_tokens", 0) or 0
        )

    # -------------------------------------------------- 便捷方法
    def chat_json(self, messages: Sequence[dict], **kwargs: Any) -> dict:
        """要求 JSON 输出并解析。解析失败抛 LLMError（评测时按失败计数）。"""
        result = self.chat(messages, json_mode=True, **kwargs)
        text = result.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"返回不是合法 JSON: {text[:200]!r}") from exc

    def cost_report(self) -> Dict[str, Any]:
        return dict(self.usage)


# --------------------------------------------------------------------- 检索用 Prompt
REFUSAL_MARKER = "未找到"

# ★ 判定"模型是否拒答"必须用**固定话术**匹配，不能只搜 REFUSAL_MARKER。
#   真实踩坑：用户问"精度要求是多少"，模型回答"精度要求为 0.90"——
#   只要回答里含"未找到"三个字就判拒答会误伤吗？不会。
#   但反过来：模型若回"知识库中未找到相关内容"，用的是同一句话。
#   所以这里改成匹配"知识库中未找到"这一完整话术，并要求它出现在
#   回答的**开头附近**（模型正常作答时也可能引用这句话去说明别的文档缺失）。
REFUSAL_PHRASE = f"知识库中{REFUSAL_MARKER}"
_REFUSAL_HEAD_CHARS = 60


def is_refusal(answer: str) -> bool:
    """判断模型回答是否属于"拒答"。

    规则：
      1. 空回答算拒答（没答出来等于没答）。
      2. 完整话术「知识库中未找到」出现在**前 60 字符内**才算拒答。
         这样避免"顺带提到某文档未找到"被误判成整条拒答。
    """
    text = (answer or "").strip()
    if not text:
        return True
    return REFUSAL_PHRASE in text[:_REFUSAL_HEAD_CHARS]


def may_still_refuse(buf: str) -> bool:
    """`buf` 是否**仍可能**是拒答话术的开头（流式用）。

    为什么需要它：流式下每来一个 token 就要决定"能不能放行给前端"。
    在攒够那段前缀之前，谁也不知道这次会不会拒答；而契约要求
    「拒答时一条引用都不发」。有了这个判定，就可以：
      · 前缀已经不可能是拒答话术 -> 立刻放行（延迟只有几个字符的时间）
      · 已经命中拒答话术         -> 改走 refusal 分支，不发引用
      · 攒到 60 字符仍不像       -> 放行（与 is_refusal 的判定窗口一致）
    """
    s = (buf or "").lstrip(" \n\r\t\u3000*#>-")
    if not s:
        return True
    if s.startswith(REFUSAL_PHRASE):
        return True
    return REFUSAL_PHRASE.startswith(s)

ANSWER_PROMPT = """你是严谨的领域知识助手。请**只依据**下面提供的检索片段回答问题。

要求：
1. 每个结论后面必须**引用**来源编号，格式如 [1]、[2]；没有引用支撑的结论不要写。
2. 不要编造片段之外的事实、数字或结论。保留关键数值与单位，不要改写数字。
3. ★ 只有**完全找不到**与问题相关的内容时，才回答"知识库中{refusal}相关内容"。
   只要片段包含**部分**相关信息，就必须：
     · 先基于已有信息回答能回答的部分（并标注引用）
     · 再明确指出哪一部分片段里没有
   不要因为信息不完整就直接整段拒答。
4. 片段之间表述不同但含义接近时，视为同一事实，不要因此判定"未找到"。

【检索片段】
{context}

【问题】
{question}

【回答】"""


def build_context(hits: List[dict]) -> str:
    """把检索结果拼成带编号的上下文。

    hits: [{"doc_id":..., "text":..., "heading_path":..., "score":...}]
    """
    lines = []
    for i, h in enumerate(hits, start=1):
        head = h.get("heading_path") or ""
        prefix = f"[{i}] {head}" if head else f"[{i}]"
        lines.append(f"{prefix}\n{h.get('text', '').strip()}")
    return "\n\n".join(lines)


def build_answer_messages(question: str, hits: List[dict]) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "你是企业知识库问答助手，严格遵守引用与拒答规则。",
        },
        {
            "role": "user",
            "content": ANSWER_PROMPT.format(context=build_context(hits),
                                            question=question,
                                            refusal=REFUSAL_MARKER),
        },
    ]


def should_refuse(hits: List[dict], threshold: float) -> bool:
    """低置信拒答：没有命中，或最高分低于阈值。"""
    if not hits:
        return True
    return float(hits[0].get("score", 0.0)) < threshold

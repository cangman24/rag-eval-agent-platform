# -*- coding: utf-8 -*-
"""core/mcp.py —— MCP 风格的工具定义与注册中心。

对应简历里那句：「按 MCP 规范将原子能力封装为标准化工具，
建立工具注册中心与 Schema 自动转换」。

这里只放**协议与注册表**（零外部依赖，可离线单测）；
具体工具实现放在 `rag2_kb_agent/tools.py`，运行时注册进来。
这样 core 不会依赖任何上层业务模块。

MCP 工具的三个核心字段（对齐 Model Context Protocol 的 tools 概念）：
    name        工具唯一标识
    description 给模型看的自然语言说明（决定模型会不会选对工具）
    inputSchema 参数的 JSON Schema（决定模型能不能填对参数）
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


class ToolError(Exception):
    """工具执行失败（会被引擎捕获并转成观察结果，而不是让整轮崩掉）。"""


class ToolNotFound(ToolError):
    pass


class ToolNotAllowed(ToolError):
    """工具不在当前意图的白名单里 —— 越权调用的拦截点。"""


class InvalidArguments(ToolError):
    """参数不符合 inputSchema。"""


# --------------------------------------------------------------------- Schema
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for_annotation(ann: Any) -> Dict[str, Any]:
    """把 Python 类型注解映射成 JSON Schema 片段。"""
    origin = getattr(ann, "__origin__", None)
    if origin in (list, List):
        args = getattr(ann, "__args__", ())
        item = _schema_for_annotation(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin in (dict, Dict):
        return {"type": "object"}
    if ann in _TYPE_MAP:
        return {"type": _TYPE_MAP[ann]}
    return {"type": "string"}


# --------------------------------------------------------------------- PEP 563
# ★ 踩坑记录（2026-10-25，接 MCP 协议层时抓到的潜伏 bug）：
#
#   模块一旦写了 `from __future__ import annotations`（PEP 563），
#   `inspect.signature` 拿到**字符串**注解（"int" 而不是 int）。
#   `_schema_for_annotation("int")` 查不到 _TYPE_MAP，落入默认
#   {"type": "string"} —— 于是 rag2_kb_agent/app.py 里
#   `kb_search(query: str, top_k: int = 3)` 生成的 schema 把 top_k
#   标成 string，模型传 top_k=5（整数）会被 validate_against_schema
#   拒绝："参数 top_k 类型应为 string，实际是 int"。
#   即：**带默认值的 int/float/bool/list 参数全线不可用**，
#   而 str 参数因为默认值恰好是 string 而看起来正常——最难查的那类。
#
#   修复：build_input_schema 前先把字符串注解解析回类型。
_STRING_ANNOTATION_TYPES = {
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "List": list, "Dict": dict, "Tuple": tuple, "Set": set,
    "Sequence": list, "Iterable": list, "Mapping": dict,
    "Any": str,          # Any 无法表达，按最宽松的 string 处理
}


def _resolve_annotation(ann: Any) -> Any:
    """PEP 563 字符串注解 -> 类型；已是类型则原样返回。"""
    if not isinstance(ann, str):
        return ann
    s = ann.strip()
    optional = re.fullmatch(r"(?:Optional|typing\.Optional)\[(.+)\]", s)
    if optional:
        s = optional.group(1).strip()
    base = s.split("[", 1)[0].strip()
    return _STRING_ANNOTATION_TYPES.get(base, str)


def build_input_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """从函数签名 + docstring 自动生成 JSON Schema。

    约定：docstring 里形如 `参数名: 说明` 的行会被收进参数描述 ——
    这段描述会直接进入模型看到的 prompt，写得好坏直接影响工具选中率。
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    param_docs: Dict[str, str] = {}
    for line in doc.splitlines():
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lstrip("-*").strip()
            if key and " " not in key:
                param_docs[key] = val.strip()

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            continue
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
        fragment = _schema_for_annotation(_resolve_annotation(ann))
        if name in param_docs:
            fragment["description"] = param_docs[name]
        if param.default is not inspect.Parameter.empty:
            fragment["default"] = param.default
        else:
            required.append(name)
        properties[name] = fragment

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def json_type_ok(value: Any, json_type: str) -> bool:
    """JSON Schema 基础类型校验（bool 要排在 int 前面，否则 True 会被当成 integer）。"""
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


def validate_against_schema(args: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """按 schema 校验参数，返回错误清单（空列表表示通过）。"""
    errors: List[str] = []
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    for key in required:
        if key not in args or args[key] is None:
            errors.append(f"缺少必填参数: {key}")
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            # 不拒绝未知参数（模型偶尔会多给），但也不参与调用
            continue
        if value is None:
            continue
        expected = spec.get("type")
        if expected and not json_type_ok(value, expected):
            errors.append(f"参数 {key} 类型应为 {expected}，实际是 {type(value).__name__}")
        if expected == "array" and "items" in spec:
            item_type = spec["items"].get("type")
            if item_type:
                for i, item in enumerate(value):
                    if not json_type_ok(item, item_type):
                        errors.append(f"参数 {key}[{i}] 类型应为 {item_type}")
    return errors


# --------------------------------------------------------------------- 工具
@dataclass
class ToolSpec:
    """一个可被 Agent 调用的工具。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[..., Any]
    # 给审计/展示用：是否为有副作用的敏感工具（需 Human-in-the-Loop 确认）
    sensitive: bool = False
    timeout_s: float = 30.0
    tags: List[str] = field(default_factory=list)

    def call(self, **kwargs: Any) -> Any:
        errors = validate_against_schema(kwargs, self.input_schema)
        if errors:
            raise InvalidArguments("; ".join(errors))
        # 只传 schema 里声明过的参数，避免被模型塞进意外的关键字
        allowed = set((self.input_schema.get("properties") or {}).keys())
        clean = {k: v for k, v in kwargs.items() if k in allowed}
        try:
            return self.func(**clean)
        except ToolError:
            raise
        except TypeError as exc:
            raise InvalidArguments(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - 工具内部异常统一包装
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    def to_openai_tool(self) -> Dict[str, Any]:
        """转成 OpenAI / DeepSeek function calling 的 tools 格式。

        这一步就是简历里的「Schema 自动转换」——模型只认这个格式，
        而我们内部用 ToolSpec，转换在注册中心统一做。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_mcp_dict(self) -> Dict[str, Any]:
        """转成 MCP tools/list 的返回形态（便于以后真的接 MCP server）。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class ToolRegistry:
    """工具注册中心 + 白名单过滤。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ---------------------------------------------------------- 注册
    def register(self, tool: ToolSpec, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"工具已存在: {tool.name}")
        self._tools[tool.name] = tool

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        sensitive: bool = False,
        tags: Optional[Sequence[str]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器注册。desc/schema 自动从函数签名与 docstring 提取。

        用法：
            @registry.tool(description="检索知识库")
            def kb_search(query: str, top_k: int = 5):
                '''query: 检索词
                top_k: 返回条数'''
                ...
        """

        def deco(func: Callable[..., Any]) -> Callable[..., Any]:
            tname = name or func.__name__
            tdesc = description or (inspect.getdoc(func) or "").split("\n\n")[0].strip()
            self.register(ToolSpec(
                name=tname,
                description=tdesc,
                input_schema=build_input_schema(func),
                func=func,
                sensitive=sensitive,
                tags=list(tags or []),
            ))
            return func

        return deco

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ---------------------------------------------------------- 查询
    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolNotFound(f"未注册的工具: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools)

    def all(self) -> List[ToolSpec]:
        return [self._tools[n] for n in self.names()]

    def __len__(self) -> int:
        return len(self._tools)

    # ---------------------------------------------------------- 白名单
    def resolve_allowed(self, allowed: Optional[Iterable[str]]) -> List[ToolSpec]:
        """把白名单名字解析成 ToolSpec 列表。

        ★ 这是「工具白名单」的落地点：意图决定 task_type，
          task_type 决定 allowed，模型只能在这个子集里选工具。
        """
        if allowed is None:
            return self.all()
        out: List[ToolSpec] = []
        for name in allowed:
            if name in self._tools:
                out.append(self._tools[name])
            # 白名单里出现未注册的名字：忽略而不是报错，
            # 否则一个配置笔误会让整轮 Agent 直接挂掉
        return out

    def openai_tools(self, allowed: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self.resolve_allowed(allowed)]

    def mcp_tools(self, allowed: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        return [t.to_mcp_dict() for t in self.resolve_allowed(allowed)]

    # ---------------------------------------------------------- 调用
    def call(self, name: str, args: Optional[Dict[str, Any]] = None,
             allowed: Optional[Iterable[str]] = None) -> Any:
        """带白名单校验的调用入口。

        白名单非 None 且 name 不在其中 -> ToolNotAllowed。
        这是拦截越权调用的唯一关口，**不要绕过它直接调 ToolSpec.call**。
        """
        tool = self.get(name)
        if allowed is not None and name not in set(allowed):
            raise ToolNotAllowed(f"工具 {name} 不在当前任务的白名单内")
        return tool.call(**(args or {}))

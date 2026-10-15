# -*- coding: utf-8 -*-
"""core/loaders.py —— 多格式文档解析。

对应简历里的「设计统一文档入库 pipeline，将 PDF、Markdown、JSON、TXT
等多格式文档标准化为可检索知识单元」。

★ 设计要点（三个真实工程问题）：

  1. **依赖是"可选"的**：解析 PDF 需要 pypdf、解析 DOCX 需要 python-docx，
     这些库可能装不上（本项目环境就没装 pypdf）。所以**延迟导入 + 清晰的
     错误信息**，而不是在模块顶层 import 让整个包挂掉。
     装不上 PDF 解析器时，其它格式照常工作 —— 这在真实项目里是必须的。

  2. **解析失败不能中断整批入库**：一个坏文件不该让 500 篇文档的入库任务失败。
     所以提供 `load_dir(..., on_error="skip")` 与逐文件错误收集。

  3. **保留结构信息**：DOCX 要按段落+表格导出并保留标题层级
     （与 `retrieval.recursive_chunk` 的 heading 识别配合），
     PDF 要保留页码（便于引用时定位"第 N 页"）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .text import normalize_whitespace


# --------------------------------------------------------------------- 结构
@dataclass
class LoadedDoc:
    """一篇解析完成的文档。"""

    doc_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "text": self.text,
                "source": self.source, "metadata": self.metadata}


class LoaderError(Exception):
    """解析失败。"""


class UnsupportedFormat(LoaderError):
    pass


class MissingDependency(LoaderError):
    """缺少解析该格式所需的第三方库。"""


# --------------------------------------------------------------------- 注册
_SUFFIX_HANDLERS: Dict[str, str] = {
    ".txt": "text",
    ".text": "text",
    ".log": "text",
    ".csv": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "markdown",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".docx": "docx",
    ".doc": "doc",          # 老格式，python-docx 读不了，给出明确提示
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
}


def supported_suffixes() -> List[str]:
    return sorted(_SUFFIX_HANDLERS)


def detect_format(path: str) -> str:
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in _SUFFIX_HANDLERS:
        raise UnsupportedFormat(
            f"不支持的格式 {suffix!r}；已支持: {', '.join(supported_suffixes())}"
        )
    return _SUFFIX_HANDLERS[suffix]


def _doc_id_for(path: str, root: Optional[str] = None) -> str:
    """由路径生成稳定 doc_id（同一文件多次入库得到同一个 id）。"""
    if root:
        try:
            rel = os.path.relpath(path, root)
        except ValueError:
            rel = path
    else:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


# --------------------------------------------------------------------- 各格式
def load_text(path: str, encoding: str = "utf-8") -> str:
    """纯文本。自动处理 BOM 与常见编码回退。"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in (encoding, "utf-8-sig", "gb18030", "utf-16", "latin-1"):
        try:
            return normalize_whitespace(raw.decode(enc))
        except (UnicodeDecodeError, LookupError):
            continue
    return normalize_whitespace(raw.decode("utf-8", errors="replace"))


def load_markdown(path: str) -> str:
    """Markdown：保留标题（`#` 开头行），去掉代码围栏语言标记噪声。

    ★ 为什么保留 `#`：`retrieval.recursive_chunk` 靠 `#` 数量判断标题层级，
      剥掉就丢失了 heading_path 的依据。
    """
    text = load_text(path)
    out: List[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # 去掉图片/链接语法，保留可见文字
        if stripped.startswith("!["):
            continue
        out.append(line)
    return normalize_whitespace("\n".join(out))


def load_json(path: str) -> str:
    """JSON：递归把值抽成可读文本行。

    ★ 为什么不直接 str(json)：原始 JSON 的括号引号会污染分词，
      让 BM25 收到一堆 `{`、`"` 噪声 token。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines: List[str] = []

    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{prefix}{i}.")
        elif node is None:
            return
        else:
            key = prefix.rstrip(".")
            lines.append(f"{key}: {node}" if key else str(node))

    walk(data)
    return normalize_whitespace("\n".join(lines))


def load_jsonl(path: str) -> str:
    """JSONL：逐行解析；坏行跳过并计数（不中断）。"""
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                lines.append("；".join(f"{k}: {v}" for k, v in obj.items()))
            else:
                lines.append(str(obj))
    return normalize_whitespace("\n".join(lines))


def load_docx(path: str) -> str:
    """DOCX：按文档顺序导出段落与表格，保留标题样式对应的层级标记。

    ★ 保留层级的方式：把 Word 的 `Heading N` 样式转成等量 `#`，
      这样 `recursive_chunk` 的 heading 识别逻辑可以直接复用，不必为
      DOCX 写第二套切分规则。
    """
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise MissingDependency("解析 .docx 需要 python-docx：pip install python-docx") from exc

    document = docx.Document(path)
    lines: List[str] = []

    def heading_prefix(style_name: str) -> Optional[str]:
        name = (style_name or "").strip()
        # 英文模板 Heading 1 / 中文模板 标题 1
        for token in ("heading", "标题"):
            if token in name.lower():
                digits = "".join(ch for ch in name if ch.isdigit())
                level = int(digits) if digits else 1
                return "#" * max(1, min(level, 6)) + " "
        return None

    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        prefix = heading_prefix(getattr(para.style, "name", ""))
        lines.append(f"{prefix}{text}" if prefix else text)

    # 表格单独导出为 Markdown 表，保留行列结构（数值型内容常在表里）
    for ti, table in enumerate(document.tables):
        lines.append(f"### 表格 {ti + 1}")
        for row in table.rows:
            cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")

    return normalize_whitespace("\n".join(lines))


def load_pdf(path: str, max_pages: Optional[int] = None) -> str:
    """PDF：逐页抽取文本，页间插入页码标记便于引用定位。

    依赖 pypdf（延迟导入）。若未安装，抛 MissingDependency 并给出安装命令。
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise MissingDependency(
                "解析 .pdf 需要 pypdf：pip install pypdf\n"
                "（若装不上，可先用 pdfplumber / 外部工具转成 txt，本模块照常处理 .txt）"
            ) from exc

    reader = PdfReader(path)
    pages: List[str] = []
    total = len(reader.pages)
    limit = min(total, max_pages) if max_pages else total
    for i in range(limit):
        try:
            content = reader.pages[i].extract_text() or ""
        except Exception:  # noqa: BLE001 - 单页坏掉不该让整篇失败
            content = ""
        if content.strip():
            pages.append(f"[第 {i + 1} 页]\n{content}")
    return normalize_whitespace("\n\n".join(pages))


def load_html(path: str) -> str:
    """HTML：用标准库剥标签（不引入 bs4）。

    ★ 取舍：html.parser 对畸形 HTML 不如 bs4 健壮，但零依赖。
      真实项目若 HTML 复杂，换成 bs4/trafilatura 只需替换本函数。
    """
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        SKIP = {"script", "style", "noscript", "head"}
        BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: List[str] = []
            self._skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP:
                self._skip_depth += 1
            elif tag in self.BLOCK:
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in self.SKIP and self._skip_depth:
                self._skip_depth -= 1
            elif tag in self.BLOCK:
                self.parts.append("\n")

        def handle_data(self, data):
            if not self._skip_depth and data.strip():
                self.parts.append(data.strip() + " ")

    with open(path, "rb") as f:
        raw = f.read()
    text = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    stripper = _Stripper()
    stripper.feed(text or raw.decode("utf-8", errors="replace"))
    return normalize_whitespace("".join(stripper.parts))


def load_doc(path: str) -> str:
    """老版 .doc：python-docx 不支持，给出可执行的替代方案。"""
    raise MissingDependency(
        ".doc（1997-2003 二进制格式）无法用 python-docx 解析。"
        "请先转换：LibreOffice `soffice --convert-to docx`，"
        "或另存为 docx/txt 后重试。"
    )


_HANDLERS: Dict[str, Callable[[str], str]] = {
    "text": load_text,
    "markdown": load_markdown,
    "json": load_json,
    "jsonl": load_jsonl,
    "docx": load_docx,
    "doc": load_doc,
    "pdf": load_pdf,
    "html": load_html,
}


# --------------------------------------------------------------------- 入口
def load_file(path: str, root: Optional[str] = None,
              fmt: Optional[str] = None) -> LoadedDoc:
    """解析单个文件 -> LoadedDoc。"""
    if not os.path.isfile(path):
        raise LoaderError(f"文件不存在: {path}")
    fmt = fmt or detect_format(path)
    handler = _HANDLERS.get(fmt)
    if handler is None:
        raise UnsupportedFormat(f"未注册的格式处理器: {fmt}")

    text = handler(path)
    doc_id = _doc_id_for(path, root)
    return LoadedDoc(
        doc_id=doc_id,
        text=text,
        source=os.path.abspath(path),
        metadata={
            "format": fmt,
            "filename": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "char_len": len(text),
        },
    )


def load_dir(
    path: str,
    recursive: bool = True,
    on_error: str = "skip",
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """批量解析目录。

    Args:
        on_error: "skip"（默认，收集错误继续）| "raise"（遇到第一个错误就抛）
        include/exclude: 后缀过滤，如 include=[".md", ".txt"]

    Returns:
        {"docs": [LoadedDoc...], "errors": [{"path":..., "error":...}],
         "skipped": [...]}
    """
    if on_error not in ("skip", "raise"):
        raise ValueError("on_error 只能是 'skip' 或 'raise'")
    if not os.path.isdir(path):
        raise LoaderError(f"目录不存在: {path}")

    include = [s.lower() for s in include] if include else None
    exclude = [s.lower() for s in exclude] if exclude else None

    docs: List[LoadedDoc] = []
    errors: List[Dict[str, str]] = []
    skipped: List[str] = []

    walker: Iterable[str]
    if recursive:
        def gen():
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in filenames:
                    yield os.path.join(dirpath, name)
        walker = gen()
    else:
        walker = (os.path.join(path, n) for n in sorted(os.listdir(path))
                  if os.path.isfile(os.path.join(path, n)))

    for fp in walker:
        suffix = os.path.splitext(fp)[1].lower()
        if include and suffix not in include:
            skipped.append(fp)
            continue
        if exclude and suffix in exclude:
            skipped.append(fp)
            continue
        if suffix not in _SUFFIX_HANDLERS:
            skipped.append(fp)
            continue
        try:
            docs.append(load_file(fp, root=path))
        except LoaderError as exc:
            if on_error == "raise":
                raise
            errors.append({"path": fp, "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001 - 兜底，保证批量任务不中断
            if on_error == "raise":
                raise
            errors.append({"path": fp, "error": f"{type(exc).__name__}: {exc}"})

    return {"docs": docs, "errors": errors, "skipped": skipped}


def summarize_load(result: Dict[str, Any]) -> str:
    """把 load_dir 的结果渲染成一行摘要（日志/报告用）。"""
    docs = result.get("docs") or []
    errors = result.get("errors") or []
    skipped = result.get("skipped") or []
    chars = sum(len(d.text) for d in docs)
    return (f"入库 {len(docs)} 篇 / 共 {chars} 字符；"
            f"失败 {len(errors)} 篇；跳过 {len(skipped)} 个文件")

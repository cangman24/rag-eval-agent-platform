# -*- coding: utf-8 -*-
"""core/store.py —— 对话持久化（SQLite，纯标准库）。

★ 为什么是 SQLite 而不是 Redis / Postgres：
      - 零依赖：`sqlite3` 在 CPython 标准库里，CI 的零依赖环境也能跑真库测试，
        不需要 mock 一层 ORM（这正好是本项目"零依赖可测"不变量的延伸）；
      - 单机单文件：`data/app.db` 可以直接拷走、可以直接用 DB Browser 打开给面试官看；
      - 事务语义足够：一次问答 = 1 条 user + 1 条 assistant + n 条 citation，
        要么全写要么全不写，刷新页面不会看到"有问无答"的半截记录。

★ 表设计（三层，引用单独成表而不是塞 JSON）：
     conversations  会话（一个浏览器标签页 / 一次连续对话）
     messages       消息（role=user|assistant，拒答也落库，refused 标记）
     citations      出处（message_id + idx 主键，可按 doc_id 聚合统计）

  引用单独成表的动机很实际：
    ① 前端刷新后要重建"出处卡片"，需要每条引用的 chunk_id 与 snippet；
    ② 简历要讲"引用可溯源"，那就得能 SQL 查「这个回答引了哪几篇文档」；
    ③ 塞进 messages.meta 的 JSON 里也能用，但 `GROUP BY doc_id` 就得在 Python 里做。

★ 线程安全：FastAPI 默认把同步端点丢线程池，所以连接用
  `check_same_thread=False` + 一把 `RLock` 串行化。写少读多，够用且不会踩
  "database is locked"（另开 WAL 让读写不互相阻塞）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------- 表结构
SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    refused         INTEGER NOT NULL DEFAULT 0,
    latency_ms      REAL,
    request_id      TEXT,
    created_at      REAL NOT NULL,
    meta            TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS citations (
    message_id TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    chunk_id   TEXT,
    doc_id     TEXT,
    score      REAL,
    snippet    TEXT,
    PRIMARY KEY (message_id, idx)
);

CREATE INDEX IF NOT EXISTS ix_messages_conv
    ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS ix_citations_doc
    ON citations(doc_id);
CREATE INDEX IF NOT EXISTS ix_conv_updated
    ON conversations(updated_at DESC);
"""

# 会话列表里做预览用的截断长度
_PREVIEW_CHARS = 80


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


class ChatStore:
    """对话存储。`path=":memory:"` 时可当纯内存实现用于测试。"""

    def __init__(self, path: str = "data/app.db") -> None:
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL：读不阻塞写；memory 库不支持，忽略失败
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------ 会话
    def create_conversation(self, title: str = "",
                            conversation_id: Optional[str] = None) -> str:
        cid = conversation_id or _new_id("c")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations(id, title, created_at, updated_at)"
                " VALUES (?,?,?,?)",
                (cid, (title or "").strip()[:_PREVIEW_CHARS], ts, ts),
            )
            self._conn.commit()
        return cid

    def ensure_conversation(self, conversation_id: Optional[str],
                            title: str = "") -> str:
        """前端传了 session_id 就直接用；不存在则建（防止刷新后 404）。"""
        if not conversation_id:
            return self.create_conversation(title)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
        if row is None:
            return self.create_conversation(title, conversation_id)
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            if row is None:
                return None
            conv = _row_to_dict(row)
            msgs = self._conn.execute(
                "SELECT * FROM messages WHERE conversation_id=?"
                " ORDER BY created_at ASC, rowid ASC",
                (conversation_id,),
            ).fetchall()
            out = []
            for m in msgs:
                item = _row_to_dict(m)
                item["refused"] = bool(item["refused"])
                try:
                    item["meta"] = json.loads(item["meta"] or "{}")
                except json.JSONDecodeError:
                    item["meta"] = {}
                item["citations"] = self._citations_for(m["id"])
                out.append(item)
            conv["messages"] = out
            conv["message_count"] = len(out)
        return conv

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """按最近更新时间倒序。带消息数 + 最后一条消息预览（侧栏用）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.*,"
                " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id)"
                "   AS message_count"
                " FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
            out = []
            for r in rows:
                item = _row_to_dict(r)
                last = self._conn.execute(
                    "SELECT content, role, refused FROM messages"
                    " WHERE conversation_id=? ORDER BY created_at DESC, rowid DESC"
                    " LIMIT 1",
                    (item["id"],),
                ).fetchone()
                preview = ""
                if last is not None:
                    text = (last["content"] or "").replace("\n", " ").strip()
                    preview = text[:_PREVIEW_CHARS]
                item["preview"] = preview
                item["has_messages"] = item["message_count"] > 0
                out.append(item)
        return out

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                ((title or "").strip()[:_PREVIEW_CHARS], _now(), conversation_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM messages WHERE conversation_id=?", (conversation_id,)
            ).fetchall()]
            for mid in ids:
                self._conn.execute("DELETE FROM citations WHERE message_id=?", (mid,))
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE id=?", (conversation_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------ 消息
    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        refused: bool = False,
        latency_ms: Optional[float] = None,
        request_id: Optional[str] = None,
        citations: Optional[Sequence[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """写一条消息（连同它的引用），并刷新会话 updated_at。"""
        cid = self.ensure_conversation(conversation_id)
        mid = _new_id("m")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(id, conversation_id, role, content, refused,"
                " latency_ms, request_id, created_at, meta)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (mid, cid, role, content or "", int(bool(refused)),
                 latency_ms, request_id, ts,
                 json.dumps(meta or {}, ensure_ascii=False, default=str)),
            )
            for i, c in enumerate(citations or [], start=1):
                chunk_id = c.get("chunk_id")
                # citation 事件里只有 chunk_id（`doc.md#p0#c0`），doc_id 得派生。
                # 不派生的话 stats 里"被引文档"永远是 0，简历上讲不了"引用覆盖了多少篇"。
                doc_id = c.get("doc_id") or (
                    str(chunk_id).split("#")[0] if chunk_id else None)
                self._conn.execute(
                    "INSERT OR REPLACE INTO citations"
                    "(message_id, idx, chunk_id, doc_id, score, snippet)"
                    " VALUES (?,?,?,?,?,?)",
                    (
                        mid,
                        int(c.get("index") or i),
                        chunk_id,
                        doc_id,
                        float(c["score"]) if c.get("score") is not None else None,
                        (c.get("snippet") or "")[:2000],
                    ),
                )
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (ts, cid))
            # 会话还没标题时，用第一个用户问题当标题（侧栏可读性）
            if role == "user":
                self._conn.execute(
                    "UPDATE conversations SET title=? WHERE id=? AND (title IS NULL OR title='')",
                    ((content or "").strip()[:_PREVIEW_CHARS], cid),
                )
            self._conn.commit()
        return mid

    def _citations_for(self, message_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT idx, chunk_id, doc_id, score, snippet FROM citations"
            " WHERE message_id=? ORDER BY idx ASC",
            (message_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------ 统计
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            conv = self._conn.execute(
                "SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
            msg = self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
            cit = self._conn.execute(
                "SELECT COUNT(*) AS n FROM citations").fetchone()["n"]
            docs = self._conn.execute(
                "SELECT COUNT(DISTINCT doc_id) AS n FROM citations "
                "WHERE doc_id IS NOT NULL").fetchone()["n"]
            top = self._conn.execute(
                "SELECT doc_id, COUNT(*) AS n FROM citations"
                " WHERE doc_id IS NOT NULL GROUP BY doc_id"
                " ORDER BY n DESC LIMIT 5").fetchall()
        return {
            "db_path": self.path,
            "conversations": conv,
            "messages": msg,
            "citations": cit,
            "distinct_docs_cited": docs,
            "top_docs": [{"doc_id": r["doc_id"], "n": r["n"]} for r in top],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------- 单例装配
_DEFAULT: Optional[ChatStore] = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> ChatStore:
    """进程级默认库：路径取 `RAG_DB_PATH`，否则 `data/app.db`。

    ★ 打不开就降级到内存库：持久化是增强功能，不能因为它让整个服务起不来。
    """
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is not None:
            return _DEFAULT
        path = os.environ.get("RAG_DB_PATH") or os.path.join("data", "app.db")
        try:
            _DEFAULT = ChatStore(path)
        except Exception:  # noqa: BLE001 - 只读挂载盘 / 权限不足
            _DEFAULT = ChatStore(":memory:")
        return _DEFAULT

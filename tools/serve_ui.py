#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""frontend/index.html 的本地启动器：静态服务 + 反向代理。

为什么需要它（而不是直接双击 html）：
  1. 页面用 fetch 打后端，file:// 打开是 null origin，会被浏览器拦掉（CORS）；
  2. 后端是 FastAPI 单一 API 服务，不提供静态文件；
  3. 让页面与 API **同源**，既不用改后端加 CORS，也不引入任何第三方依赖。

用法：
    python tools/serve_ui.py                 # 打开 http://127.0.0.1:8080
    python tools/serve_ui.py --port 9000     # 换端口
    python tools/serve_ui.py --open          # 启动后自动开浏览器

路由：
    /              -> frontend/index.html
    /<file>        -> frontend/<file>
    /api/<path>    -> 转发到 http://127.0.0.1:8000/<path>   项目二：RAG 问答
    /agent/<path>  -> 转发到 http://127.0.0.1:8100/agent/<path>  项目一：Agent 轨迹

★ 两个后端都要转发：项目二推"检索/生成阶段"事件，项目一推"Agent 推理轨迹"
  事件，是两套不同的可观测对象。前端同源地打两个前缀，各自独立。

★ 流式关键：响应不设 Content-Length（HTTP/1.0 下由连接关闭表示结束），
  并且每读到一块就 flush，SSE 帧才能逐条到达浏览器而不是憋到最后。
"""
from __future__ import annotations

import argparse
import http.client
import mimetypes
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")

_TEXT_SUFFIX = (".html", ".css", ".js", ".json", ".svg", ".txt", ".md")


class Handler(BaseHTTPRequestHandler):
    backend_host = "127.0.0.1"
    backend_port = 8000
    agent_port = 8100
    server_version = "kb-ui/0.1"
    # 保持 HTTP/1.0：不设 Content-Length 时靠 EOF 结束，天然支持流式透传

    # ------------------------------------------------------------- 路由
    def _dispatch(self, method: str) -> None:
        if self.path.startswith("/api/"):
            self._proxy(method, self.backend_port, strip="/api")
        elif self.path.startswith("/agent/"):
            self._proxy(method, self.agent_port, strip=None)
        elif method == "GET":
            self._static()
        else:
            self.send_error(405, "only /api/* and /agent/* accept %s" % method)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        # BaseHTTPRequestHandler 只默认实现 GET/HEAD/POST，其余方法不声明就是 501。
        # 会话删除（DELETE /api/sessions/{id}）走的就是这里。
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_OPTIONS(self) -> None:  # noqa: N802
        # 同源本不需要，留着方便有人把页面单独托管时联调
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods",
                         "GET,POST,PATCH,PUT,DELETE,OPTIONS")
        self.end_headers()

    # ------------------------------------------------------------- 静态
    def _static(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", ""):
            path = "/index.html"
        rel = os.path.normpath(path.lstrip("/"))

        # 防目录穿越：拼接后必须仍在 FRONTEND 之内
        full = os.path.abspath(os.path.join(FRONTEND, rel))
        if not full.startswith(os.path.abspath(FRONTEND) + os.sep) or not os.path.isfile(full):
            self.send_error(404, "not found")
            return

        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".html"):
            ctype = "text/html"
        elif full.endswith(".js"):
            ctype = "application/javascript"
        if full.endswith(_TEXT_SUFFIX):
            ctype += "; charset=utf-8"

        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------- 代理
    def _proxy(self, method: str, port: int, strip: "str | None") -> None:
        # strip="/api"   -> /api/chat/stream 发给后端的 /chat/stream
        # strip=None     -> /agent/chat/stream 原样发（后端路由本身就带 /agent）
        target = self.path[len(strip):] if strip else self.path
        # 读请求体：有 Content-Length 就读（PATCH 改标题也带 body）；
        # 无 body 的写方法显式发 0 长度，免得后端等 body 等到超时。
        raw_len = int(self.headers.get("Content-Length") or 0)
        if raw_len:
            body = self.rfile.read(raw_len)
        elif method in ("POST", "PUT", "PATCH"):
            body = b""
        else:
            body = None

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "connection", "accept-encoding", "content-length")
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))

        conn = http.client.HTTPConnection(self.backend_host, port, timeout=300)
        try:
            conn.request(method, target, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001 - 后端没起时给前端一个可读错误
            hint = ("rag1_knowledge_qa.server" if port == self.backend_port
                    else "rag2_kb_agent.agent_api")
            payload = ('{"detail":"backend unreachable: %s","hint":"%s"}'
                       % (str(exc).replace('"', "'").replace("\\", "/"),
                          hint)).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            conn.close()
            return

        self.send_response(resp.status)
        for key, value in resp.getheaders():
            if key.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(key, value)
        self.send_header("X-Accel-Buffering", "no")   # 明确告诉中间层不要缓存
        self.end_headers()

        try:
            while True:
                chunk = resp.read(512)               # 小块读 -> 小块写，保证流式
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    # ------------------------------------------------------------- 日志
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("    %s %s\n" % (self.command, self.path))


def _backend_alive(host: str, port: int, probe: str = "/health") -> bool:
    conn = http.client.HTTPConnection(host, port, timeout=2)
    try:
        conn.request("GET", probe)
        return conn.getresponse().status == 200
    except Exception:  # noqa: BLE001
        return False
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="领域知识问答 · 网页客户端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000,
                        help="项目二 RAG 问答后端")
    parser.add_argument("--agent-port", type=int, default=8100,
                        help="项目一 Agent 轨迹后端")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args(argv)

    if not os.path.isfile(os.path.join(FRONTEND, "index.html")):
        sys.stderr.write("找不到 frontend/index.html（%s）\n" % FRONTEND)
        return 1

    Handler.backend_host = args.backend_host
    Handler.backend_port = args.backend_port
    Handler.agent_port = args.agent_port

    alive = _backend_alive(args.backend_host, args.backend_port)
    alive_agent = _backend_alive(args.backend_host, args.agent_port,
                                 probe="/agent/health")
    url = "http://%s:%d/" % (args.host, args.port)

    print("网页客户端已启动：%s" % url)
    print("项目二 RAG 问答  %s:%d —— %s" % (
        args.backend_host, args.backend_port,
        "已连接" if alive else "未连接（先启动 rag1_knowledge_qa.server）"))
    print("项目一 Agent 轨迹 %s:%d —— %s" % (
        args.backend_host, args.agent_port,
        "已连接" if alive_agent
        else "未连接（先启动 uvicorn rag2_kb_agent.agent_api:app_factory --factory）"))
    print("按 Ctrl+C 停止\n")

    if args.open:
        webbrowser.open(url)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

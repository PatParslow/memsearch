"""Local web UI for the memsearch knowledge graph -- stdlib
http.server only (no Flask/FastAPI dependency, matching the project's
existing lean dependency list). Serves the static SVG map bundle plus a
small JSON REST API backed by annotations.py's SQLite store."""

from __future__ import annotations

import json
import mimetypes
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import annotations, convo_miner, graph

STATIC_DIR = Path(__file__).parent / "web"

_ROUTE_NOTE_ID = re.compile(r"^/api/notes/(\d+)$")
_ROUTE_LINK_ID = re.compile(r"^/api/links/(\d+)$")
_ROUTE_GAP_ID = re.compile(r"^/api/gaps/([0-9a-f]+)$")


def _format_conversation(path: Path) -> str:
    """Render a Claude Code .jsonl transcript as a plain-text back-and-forth
    instead of raw JSON lines -- reuses convo_miner's own parser so this
    view always matches what actually got mined/embedded."""
    messages = convo_miner.parse_session(path)
    if not messages:
        return f"(no readable user/assistant turns found in {path.name})"
    lines = [f"Claude Code conversation transcript -- {path.name}", "=" * 70, ""]
    for role, text in messages:
        lines.append(f"----- {'USER' if role == 'user' else 'ASSISTANT'} -----")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    graph_path = graph.DEFAULT_GRAPH_PATH
    db_path = annotations.DEFAULT_DB_PATH

    def log_message(self, format, *args):  # noqa: A002 -- matches base class signature
        pass  # quiet -- avoid cluttering the console with every asset request

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ---- GET ----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        elif path == "/app.css":
            self._send_file(STATIC_DIR / "app.css", "text/css; charset=utf-8")
        elif path == "/api/graph":
            self._api_get_graph()
        elif path == "/api/notes":
            node_id = qs.get("node_id", [None])[0]
            conn = annotations.get_db(self.db_path)
            data = annotations.notes_for_node(conn, node_id) if node_id else annotations.all_notes(conn)
            conn.close()
            self._send_json(data)
        elif path == "/api/links":
            conn = annotations.get_db(self.db_path)
            data = annotations.all_manual_links(conn)
            conn.close()
            self._send_json(data)
        elif path == "/api/gaps":
            conn = annotations.get_db(self.db_path)
            data = annotations.all_gaps(conn)
            conn.close()
            self._send_json(data)
        elif path == "/api/source":
            self._api_get_source(qs.get("path", [None])[0])
        elif path == "/api/synthesis-idea":
            self._api_get_synthesis_idea(qs.get("path", [None])[0], qs.get("idea", [None])[0])
        else:
            self._send_json({"error": "not found"}, 404)

    def _api_get_synthesis_idea(self, raw_path: str | None, idea_number: str | None) -> None:
        # Scoped view for a single idea within a synthesis report -- the
        # generic /api/source route (used everywhere else) always serves
        # a whole file, which is right for viewing a source file but
        # wrong here: a report can hold a dozen ideas, and a "View plan"
        # link from one specific gap should jump straight to that one,
        # not dump the whole document as unrendered text.
        if not raw_path or not idea_number:
            self._send_json({"error": "missing path or idea"}, 400)
            return
        p = Path(raw_path)
        if not p.is_file():
            self._send_json({"error": "report not found on disk"}, 404)
            return
        text = p.read_text(encoding="utf-8")
        blocks = re.split(r"(?m)^## Idea (\d+): ", text)
        idea_map = {}
        for i in range(1, len(blocks), 2):
            idea_map[blocks[i]] = blocks[i + 1]
        body = idea_map.get(idea_number)
        if body is None:
            self._send_json({"error": "idea not found in this report"}, 404)
            return
        body = re.sub(r"\n---\s*\Z", "", body.strip())
        title, _, rest = body.partition("\n")
        self._send_json({
            "report_file": p.name, "idea_number": idea_number,
            "title": title.strip(), "markdown": rest.strip(),
        })

    def _api_get_source(self, raw_path: str | None) -> None:
        if not raw_path:
            self._send_json({"error": "missing path"}, 400)
            return
        p = Path(raw_path)
        if not p.is_file():
            self._send_json({"error": "source file not found on disk"}, 404)
            return
        guessed_type, _ = mimetypes.guess_type(str(p))
        try:
            if p.suffix.lower() == ".jsonl":
                body = _format_conversation(p).encode("utf-8")
                content_type = "text/plain; charset=utf-8"
            elif guessed_type is not None and not guessed_type.startswith("text/"):
                body = p.read_bytes()
                content_type = guessed_type
            else:
                body = p.read_text(encoding="utf-8", errors="replace").encode("utf-8")
                content_type = "text/plain; charset=utf-8"
        except OSError as e:
            self._send_json({"error": str(e)}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_get_graph(self):
        try:
            data = graph.load_graph(self.graph_path)
        except FileNotFoundError:
            self._send_json({"error": "no graph built yet -- run `memsearch graph build` first"}, 404)
            return
        conn = annotations.get_db(self.db_path)
        data = dict(data)
        data["manual_links"] = annotations.all_manual_links(conn)
        data["gap_status"] = annotations.all_gaps(conn)
        conn.close()
        index_path = Path(self.graph_path).parent / "synthesis_index.json"
        try:
            data["synthesis_index"] = json.loads(index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data["synthesis_index"] = []
        self._send_json(data)

    # ---- POST ----

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_json_body()
        conn = annotations.get_db(self.db_path)
        try:
            if path == "/api/notes":
                note_id = annotations.add_note(conn, body["node_id"], body["text"])
                self._send_json({"id": note_id}, 201)
            elif path == "/api/links":
                link_id = annotations.add_manual_link(conn, body["node_a"], body["node_b"], body.get("label"))
                self._send_json({"id": link_id}, 201)
            elif path == "/api/gaps":
                gid = annotations.add_gap(
                    conn, body.get("kind", "manual"), body["node_a"], body.get("node_b"),
                    body["description"], status=body.get("status", "open"),
                )
                self._send_json({"id": gid}, 201)
            else:
                self._send_json({"error": "not found"}, 404)
        except KeyError as e:
            self._send_json({"error": f"missing field: {e}"}, 400)
        finally:
            conn.close()

    # ---- PATCH ----

    def do_PATCH(self):
        path = urlparse(self.path).path
        body = self._read_json_body()
        m = _ROUTE_GAP_ID.match(path)
        if m:
            conn = annotations.get_db(self.db_path)
            annotations.set_gap_status(conn, m.group(1), body.get("status", "open"))
            conn.close()
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, 404)

    # ---- DELETE ----

    def do_DELETE(self):
        path = urlparse(self.path).path
        conn = annotations.get_db(self.db_path)
        try:
            m = _ROUTE_NOTE_ID.match(path)
            if m:
                annotations.delete_note(conn, int(m.group(1)))
                self._send_json({"ok": True})
                return
            m = _ROUTE_LINK_ID.match(path)
            if m:
                annotations.delete_manual_link(conn, int(m.group(1)))
                self._send_json({"ok": True})
                return
            self._send_json({"error": "not found"}, 404)
        finally:
            conn.close()


def serve(port: int = 8765, graph_path: str = graph.DEFAULT_GRAPH_PATH,
          db_path: str = annotations.DEFAULT_DB_PATH, open_browser: bool = True) -> None:
    Handler.graph_path = graph_path
    Handler.db_path = db_path
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"memsearch graph UI -- {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

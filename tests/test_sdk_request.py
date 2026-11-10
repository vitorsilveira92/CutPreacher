"""Runs the real Anthropic SDK against a local server that replays a streamed Messages API response."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic

from cutpreacher.config import Settings
from cutpreacher.highlights import call_claude

CLIPS = {"clips": [{"start_seconds": 1.0, "end_seconds": 40.0, "title": "T", "hook": "h", "reason": "r",
                    "description": "d", "hashtags": ["#fé"]}]}


def sse_events(text: str) -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[:20]}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[20:]}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 50}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


def test_real_sdk_request_shape():
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["path"] = self.path
            captured["headers"] = dict(self.headers)
            captured["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = sse_events(json.dumps(CLIPS))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="sk-test", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0)
        clips = call_claude(client, Settings(), "transcrição de teste")
    finally:
        server.shutdown()

    assert clips == CLIPS["clips"]
    body = captured["body"]
    assert captured["path"].startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["stream"] is True
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "server-side-fallback-2026-07-01" in captured["headers"].get("anthropic-beta", "")

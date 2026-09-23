#!/usr/bin/env python3
"""Tiny web server for the interactive Laya Tetris demo.

Examples:
  .venv/bin/python examples/tetris/server.py --device cpu --port 8011
  .venv/bin/python examples/tetris/server.py --device npu:0 --port 8011
"""

import argparse
import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from game import TetrisGame
from policy import LayaTetrisPolicy
from runtime import DEFAULT_MODEL, load_agent

WEB_DIR = Path(__file__).resolve().parent / "static"
MAX_SESSIONS = 16


class TetrisApp:
    def __init__(self, agent):
        self.agent = agent
        self.sessions = {}
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def new_game(self, body):
        seed = body.get("seed")
        if seed is None:
            seed = random.randint(0, 99999)
        game = TetrisGame(seed=seed)
        policy = LayaTetrisPolicy(self.agent, guarded=bool(body.get("guarded", True)))
        sid = uuid.uuid4().hex[:12]
        with self.lock:
            if len(self.sessions) >= MAX_SESSIONS:
                oldest = min(self.sessions, key=lambda key: self.sessions[key]["created"])
                del self.sessions[oldest]
            self.sessions[sid] = {
                "game": game,
                "policy": policy,
                "created": time.time(),
                "stats": {"pieces": 0, "interventions": 0, "inference_ms_total": 0.0},
            }
        return {"session": sid, "seed": seed, "state": game.snapshot()}

    def step(self, session):
        with self.lock:
            entry = self.sessions.get(session)
        if entry is None:
            raise KeyError("unknown or expired tetris session")
        game, policy, stats = entry["game"], entry["policy"], entry["stats"]
        if not game.alive:
            return {"state": game.snapshot(), "stats": stats, "done": True}
        before = game.snapshot()
        with self.inference_lock:
            decision = policy.decide(game)
            executed = decision.candidates[decision.executed]
            game.apply(executed)
        after = game.snapshot()
        stats["pieces"] += 1
        stats["interventions"] += int(decision.intervened)
        stats["inference_ms_total"] += decision.inference_ms
        return {
            "decision": decision.to_dict(),
            # Board/state before this placement, used by the UI to animate the drop.
            "before": before,
            "state": after,
            "stats": stats,
            "done": not game.alive,
        }


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, fmt, *args):
        return

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status, message):
        self._send(status, {"error": message})

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/info":
            self._send(
                200,
                {
                    "model": "laya",
                    "device": str(self.app.agent.device),
                    "width": 10,
                    "height": 20,
                },
            )
        else:
            self._send_error(404, "not found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/tetris/new":
                self._send(200, self.app.new_game(body))
            elif path == "/api/tetris/step":
                session = body.get("session")
                if not session:
                    raise ValueError("session is required")
                self._send(200, self.app.step(session))
            else:
                self._send_error(404, "not found")
        except KeyError as exc:
            self._send_error(404, str(exc).strip("'"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            self._send_error(500, str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to one Laya checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, npu:0, npu:1 ...")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    agent = load_agent(args.model, args.device)
    Handler.app = TetrisApp(agent)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Tetris demo: http://%s:%d" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

"""
ControlServer — the networked transport wrapping ControlService, per
service.py's own docstring ("wrap this same class with a tiny JSON-RPC-ish
layer over WebSocket when a networked front-end is needed").

Built on FastAPI + uvicorn rather than the raw `websockets` library so the
same process/port can later mount the Stage B static web UI and docs
alongside this `/ws` route (`self.app` is a plain FastAPI instance — add
routes/mounts to it before calling `serve_in_thread()`). The wire protocol
is unaffected by this choice: `/ws` is a standard WebSocket endpoint, so any
`websockets`-based client (including this file's tests) connects to it
exactly as it would to a bare `websockets.serve` server.

Threading model (read before touching this file): uvicorn runs the FastAPI
app on its own asyncio event loop, on its own thread, started by
`serve_in_thread`. The sim loop (swap_experiment.py's `while True`) is
synchronous and lives on the main thread, calling `service.tick(obs)` once
per control step. These two must never call ControlService directly across
threads. Instead:

  - Commands: the `/ws` handler receiving `{"method": ..., "params": ...}`
    puts (websocket, message) onto `self.commands`, a plain `queue.Queue`
    (thread-safe by construction). The sim loop calls `drain_commands()`
    once per tick, which executes each queued method against ControlService
    ON THE SIM THREAD and stashes the JSON-RPC result to send back.
  - Telemetry: the sim loop calls `publish_status(service.status())` once
    per tick, which just replaces a shared dict under a lock. A background
    asyncio task broadcasts that dict to all connected clients at ~10 Hz —
    it never reads from ControlService directly.

This mirrors how viser's GUI buttons already work today (viser calls
request_switch/pause/estop from its own thread; those methods are cheap
flag-sets, which is why it's been safe) — this file just makes the
single-threaded-command-execution rule explicit instead of accidental.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# The whole ControlService surface this transport exposes. estop is always
# accepted (see dispatch()) — it must never be blocked by anything here,
# mirroring the "estop always wins" rule already enforced inside
# ControlService/SafetyGovernor.
METHODS = {"request_switch", "status", "pause", "resume", "estop"}


class ControlServer:
    def __init__(self, service, host: str = "0.0.0.0", port: int = 9013):
        self.service = service
        self.host = host
        self.port = port

        # Sim-thread -> socket-thread handoff of the latest status snapshot.
        self._status_lock = threading.Lock()
        self._latest_status: dict = {}

        # Socket-thread -> sim-thread command handoff.
        self.commands: "queue.Queue[tuple]" = queue.Queue()

        self._clients = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None

        # A plain FastAPI app exposing only /ws today. Stage B mounts the
        # static web UI and docs/ onto this SAME app (before serve_in_thread
        # is called) rather than standing up a second server/port.
        self.app = FastAPI()
        self.app.add_api_websocket_route("/ws", self._ws_endpoint)

        if host not in ("localhost", "127.0.0.1", "::1"):
            print(f"[ControlServer] WARNING: binding to non-localhost host "
                  f"'{host}' with no auth — commands (incl. request_switch, "
                  f"estop) will be reachable from the network.")

    # ---- called from the sim loop thread, once per tick ----

    def publish_status(self, status: dict) -> None:
        with self._status_lock:
            self._latest_status = status

    def drain_commands(self) -> None:
        """Execute every command queued since the last tick, on THIS
        (the sim loop's) thread — never on the socket thread, no exceptions
        (estop included: on RealAdapter, estop() is a real DDS write, and
        the whole point of this queue is that ControlService is only ever
        touched from the sim thread). Call once per tick, right alongside
        service.tick(obs). estop messages are executed first within a
        batch so they never wait behind unrelated queued commands."""
        pending = []
        while True:
            try:
                pending.append(self.commands.get_nowait())
            except queue.Empty:
                break
        pending.sort(key=lambda item: item[1].get("method") != "estop")

        for websocket, msg in pending:
            reply = self._dispatch(msg)
            self._send_threadsafe(websocket, reply)

    def _dispatch(self, msg: dict) -> dict:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method not in METHODS:
            return {"id": msg_id, "error": f"unknown method '{method}'. Valid: {sorted(METHODS)}"}

        try:
            fn = getattr(self.service, method)
            result = fn(**params)
        except Exception as e:  # noqa: BLE001 - report to the caller, don't crash the sim loop
            return {"id": msg_id, "error": f"{type(e).__name__}: {e}"}

        return {"id": msg_id, "result": result}

    # ---- socket-side (runs on the server's own thread/event loop) ----

    async def _ws_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    await websocket.send_text(json.dumps({"error": f"invalid JSON: {e}"}))
                    continue
                if not isinstance(msg, dict):
                    # Reject here, on the socket thread — never enqueue a
                    # non-dict payload. drain_commands()/dispatch() assume
                    # dict .get() access; a bad payload must not be able to
                    # crash the sim thread over a client typo.
                    await websocket.send_text(json.dumps({"error": "message must be a JSON object"}))
                    continue
                self.commands.put((websocket, msg))
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)

    def _send_threadsafe(self, websocket, payload: dict) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._safe_send(websocket, json.dumps(payload)), self._loop,
        )

    @staticmethod
    async def _safe_send(websocket, text: str) -> None:
        try:
            await websocket.send_text(text)
        except Exception:  # noqa: BLE001 - client disconnected mid-send; nothing to do
            pass

    async def _broadcast_status_loop(self) -> None:
        while True:
            await asyncio.sleep(0.1)  # ~10 Hz
            with self._status_lock:
                status = dict(self._latest_status)
            if not status or not self._clients:
                continue
            payload = json.dumps({"method": "status", "result": status})
            for ws in list(self._clients):
                await self._safe_send(ws, payload)

    def serve_in_thread(self) -> None:
        """Starts uvicorn serving `self.app` on a background thread and
        blocks until the port is actually bound (or raises if it couldn't
        be) — the sim loop keeps running on the calling thread once this
        returns. uvicorn logs a bind failure (e.g. port already in use) but
        does not raise it back to the caller, so this polls `Server.started`
        with a timeout and treats "thread died without starting" or
        "timed out" as failure, rather than letting swap_experiment.py print
        a false 'listening' message and silently drop every reply."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)

        async def _run_and_capture_loop():
            self._loop = asyncio.get_running_loop()
            await self._uvicorn_server.serve()

        def _run():
            asyncio.run(_run_and_capture_loop())

        self._thread = threading.Thread(target=_run, daemon=True, name="ControlServer")
        self._thread.start()

        for _ in range(250):  # ~5s
            if self._uvicorn_server.started:
                break
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"ControlServer failed to bind ws://{self.host}:{self.port} "
                    f"(port likely already in use — see stderr above for uvicorn's error)"
                )
            self._thread.join(timeout=0.02)
        else:
            raise RuntimeError(f"ControlServer on port {self.port} did not start within 5s")

        # The broadcast loop is independent of any single connection; start
        # it once the server's event loop is confirmed running.
        asyncio.run_coroutine_threadsafe(self._broadcast_status_loop(), self._loop)

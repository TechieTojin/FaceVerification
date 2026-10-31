"""Local kiosk web app: the kiosk UI and its local API. Bind to 127.0.0.1 only."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from kiosk.agent import KioskAgent, KioskError

log = logging.getLogger("kiosk")
MAX_UPLOAD = 2_100_000


class StartBody(BaseModel):
    user_id: str


def create_app(agent: KioskAgent, ui_dir: Optional[str], sync_interval: float) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def loop():
            while True:
                await run_in_threadpool(agent.sync, True)
                await asyncio.sleep(sync_interval)
        task = asyncio.create_task(loop()) if sync_interval > 0 else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="FaceVerification kiosk", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(KioskError)
    async def _err(_: Request, exc: KioskError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message, "code": exc.code})

    @app.middleware("http")
    async def _headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/kiosk/api/status")
    def status():
        return agent.status()

    @app.post("/kiosk/api/sync")
    def sync():
        return {**agent.sync(True), "status": agent.status()}

    @app.get("/kiosk/api/subjects")
    def subjects():
        b = agent.bundle_state()
        return {"subjects": agent.store.subjects() if b["valid"] and not agent.revoked else [], "bundle": b}

    @app.get("/kiosk/api/events")
    def events():
        return {"events": agent.recent_events(), "counts": agent.store.counts()}

    @app.post("/kiosk/api/sessions", status_code=201)
    def start(body: StartBody):
        return agent.start_session(body.user_id)

    @app.post("/kiosk/api/sessions/{session_id}/frames")
    async def frame(session_id: str, seq: int = Form(..., ge=1), frame_token: str = Form(..., max_length=128),
                    file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise KioskError(413, "FRAME_TOO_LARGE", "Frame too large")
        return await run_in_threadpool(agent.submit_frame, session_id, seq, frame_token, data)

    @app.post("/kiosk/api/sessions/{session_id}/cancel")
    def cancel(session_id: str):
        return agent.cancel(session_id)

    if ui_dir and Path(ui_dir, "index.html").exists():
        root = Path(ui_dir).resolve()

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith("api/"):          # the central API does not exist on the kiosk
                return JSONResponse(status_code=404, content={"detail": "Not available on the kiosk", "code": "KIOSK"})
            target = (root / path).resolve()
            if path and target.is_file() and root in target.parents:
                return FileResponse(target)
            return FileResponse(root / "index.html")

    return app

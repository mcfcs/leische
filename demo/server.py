"""FastAPI app for the thesis demo.

    uv run --extra demo uvicorn demo.server:app --host 127.0.0.1 --port 8000

Env:
    LEISCHE_CHECKPOINT    path to the .pt (default cache/checkpoints/ablation-8_full-fold0-seed13.pt)
    LEISCHE_RQ3_HEADS     path to the RQ3 sentiment-head pickle
    LEISCHE_DEMO_DEVICE   "cpu" | "cuda" (default: cuda if available, else cpu)

The model runs on whichever device is resolved; while the GPU is training, set
LEISCHE_DEMO_DEVICE=cpu.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine

STATIC = Path(__file__).resolve().parent / "static"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_engine: Engine | None = None


def engine() -> Engine:
    if _engine is None:                      # startup failed or has not run yet
        raise HTTPException(503, "model is still loading")
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    _engine = Engine()      # ~20 s on CPU: dataset, XLM-R, banks, examples
    yield
    _engine = None


app = FastAPI(title="Leische — context-aware sarcasm demo", docs_url="/api/docs",
              lifespan=lifespan)


# --------------------------------------------------------------- schemas
class Submission(BaseModel):
    title: str = ""
    selftext: str = ""


class Turn(BaseModel):
    text: str = ""
    is_submitter: bool = False


class HistoryItem(BaseModel):
    text: str = ""
    hours_ago: float = 1.0


class PredictRequest(BaseModel):
    text: str
    submission: Submission | None = None
    parents: list[Turn] = Field(default_factory=list)
    replies: list[Turn] = Field(default_factory=list)
    history: list[HistoryItem] = Field(default_factory=list)
    use_conv: bool = True
    use_temp: bool = True
    use_ret: bool = True
    exclude_thread: str | None = None


# ---------------------------------------------------------------- routes
@app.get("/api/meta")
def api_meta() -> dict:
    return engine().meta()


@app.get("/api/examples")
def api_examples() -> list[dict]:
    return engine().examples


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> dict:
    try:
        return engine().predict(req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

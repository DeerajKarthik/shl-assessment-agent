from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.parent

from app.schemas import ChatRequest, ChatResponse
from app.service import RecommenderService
from app.settings import settings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = RecommenderService(settings)
    logger.info(
        "recommender ready: catalog=%s model_enabled=%s",
        len(app.state.recommender.catalog.items),
        app.state.recommender.gemini.enabled,
    )
    yield


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = ROOT / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest, response: Response) -> ChatResponse:
    service: RecommenderService = request.app.state.recommender
    res = await service.chat(payload)
    if hasattr(res, 'provider') and res.provider:
        response.headers["X-Model-Provider"] = res.provider
    return res


import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ask_the_lion.config import Settings
from ask_the_lion.rag import RagPipeline, build_pipeline

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    # Length is enforced by the input guardrail so clients get a `too_long` refusal.
    question: str


class SourceOut(BaseModel):
    question_number: int
    page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    refused: bool
    reason: str | None


def create_app(pipeline: RagPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_pipeline(settings)
        yield

    app = FastAPI(title="Ask the Lion", lifespan=lifespan)
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["POST", "GET"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_body_bytes:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    @limiter.limit(settings.rate_limit)
    def ask(request: Request, body: AskRequest) -> AskResponse:
        result = request.app.state.pipeline.ask(body.question)
        return AskResponse(
            answer=result.answer,
            sources=[SourceOut(**vars(s)) for s in result.sources],
            refused=result.refused,
            reason=result.reason.value if result.reason else None,
        )

    return app

"""The scoring service.

Run it:      uv run uvicorn api.main:app --reload
Interactive: http://127.0.0.1:8000/docs
"""

import json
import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fraud_radar.scoring import Scorer

from .schemas import HealthResponse, ScoreResponse, Transaction


class JsonFormatter(logging.Formatter):
    """One JSON object per line.

    Structured rather than printed prose, because in Phase 4 these lines get
    parsed to compute throughput and p50/p95 latency. `print()` output would
    have to be regex-scraped; JSON can just be read.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("fraud_radar.api")
log.setLevel(logging.INFO)
log.addHandler(handler)
log.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup.

    Loading it per request would add tens of milliseconds to every score and
    re-read the file thousands of times. It is immutable, so one copy is shared
    across all requests.
    """
    app.state.scorer = Scorer()
    log.info(
        "model loaded",
        extra={
            "extra_fields": {
                "model": app.state.scorer.model_name,
                "version": app.state.scorer.version,
                "threshold": round(app.state.scorer.threshold, 6),
            }
        },
    )
    yield
    log.info("shutting down")


app = FastAPI(
    title="Fraud Radar",
    description="Real-time transaction risk scoring.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Turn Pydantic's output into something a caller can act on.

    The default response is a nested structure that is awkward to read. This
    flattens it to 'which field, and what was wrong with it'.
    """
    problems = [
        {"field": ".".join(str(p) for p in err["loc"] if p != "body"),
         "problem": err["msg"]}
        for err in exc.errors()
    ]
    log.warning("rejected invalid payload",
                extra={"extra_fields": {"problems": problems}})
    return JSONResponse(
        status_code=422,
        content={"error": "invalid transaction", "problems": problems},
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness check. Phase 3's docker-compose waits on this before starting."""
    scorer = request.app.state.scorer
    return HealthResponse(
        status="ok",
        model_version=scorer.version,
        model_name=scorer.model_name,
        cards_tracked=len(scorer.state),
    )


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, request: Request) -> ScoreResponse:
    """Score one transaction and fold it into that card's running history."""
    scorer = request.app.state.scorer
    started = time.perf_counter()

    result = scorer.score(txn.model_dump())

    latency_ms = (time.perf_counter() - started) * 1000
    log.info(
        "scored",
        extra={
            "extra_fields": {
                "cc_num_suffix": str(txn.cc_num)[-4:],  # never log a full card number
                "amt": txn.amt,
                "category": txn.category,
                "risk_score": round(result["risk_score"], 6),
                "flagged": result["flagged"],
                "latency_ms": round(latency_ms, 2),
            }
        },
    )
    return ScoreResponse(**result, latency_ms=round(latency_ms, 3))

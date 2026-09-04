# One image, three roles: the API, the consumer, and the producer all run the
# same code and differ only in their command. Building it once keeps them on
# identical dependencies — a consumer scoring with a different scikit-learn
# version than the API would produce subtly different numbers.
FROM python:3.11-slim

# uv is copied from its own published image rather than curl'd, so the build
# does not depend on a network installer or on the host's certificate store.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer. Application code changes constantly;
# the lockfile rarely does, so this layer stays cached across most rebuilds.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY fraud_radar/ ./fraud_radar/
COPY api/ ./api/
COPY streaming/ ./streaming/
COPY scripts/ ./scripts/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

"""FastAPI application: the web layer only.

Parses the request, runs the pipeline, and maps failures to HTTP responses.
"""

import logging

from fastapi import Depends, FastAPI, HTTPException

from app.errors import LLMError
from app.extraction import run_extraction
from app.llm import LLMFunc, extract_raw
from app.schemas import ExtractRequest, ExtractResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="V-Vet Symptom Extractor",
    description="Turns a farmer's description of a sick animal into structured findings.",
)


def get_llm() -> LLMFunc:
    """Dependency: provides the function that calls the LLM.

    The endpoint asks for this via Depends(). Tests replace it with a fake
    using app.dependency_overrides, so no network or API key is needed.
    """
    return extract_raw


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, llm: LLMFunc = Depends(get_llm)) -> ExtractResponse:
    """Extract structured findings from a farmer's message."""
    try:
        return run_extraction(request.message, llm)
    except LLMError as exc:
        # Log the real cause for us; give the caller a safe, generic message.
        logger.error("Extraction failed: %s", exc)
        # 503, NOT a response with escalate=false: when we can't tell, we say so.
        raise HTTPException(
            status_code=503,
            detail="Extraction is temporarily unavailable. Please try again.",
        ) from exc
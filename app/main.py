"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="V-Vet Symptom Extractor")

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: confirms the server is up."""
    return {"status": "ok"}

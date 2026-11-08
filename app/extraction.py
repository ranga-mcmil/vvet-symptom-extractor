"""Turn raw LLM output into trusted findings, then run the whole pipeline.

Pipeline:  LLM  ->  validate/clean (this file)  ->  escalation rules
The LLM is the only untrusted step. Everything after it is plain, tested code.
"""

import logging
from typing import Any

from pydantic import ValidationError

from app.errors import LLMError
from app.escalation import decide_escalation
from app.llm import LLMFunc
from app.schemas import ExtractResponse, Finding, SignId, Value

logger = logging.getLogger(__name__)

# Higher number = more cautious. Used to resolve duplicate findings.
CAUTION_RANK = {Value.NO: 0, Value.NOT_SURE: 1, Value.YES: 2}


def _normalise(text: str) -> str:
    """Make text comparable: lowercase, one-space whitespace, straight apostrophes"""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(text.lower().split())


def clean_findings(raw: Any, message: str) -> list[Finding]:
    """Validate raw model output against the schema and the original message."""
    items = raw.get("findings") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise LLMError("Model output has no 'findings' list")

    normalised_message = _normalise(message)
    best: dict[SignId, Finding] = {}

    for item in items:
        # 1. Schema check: valid sign_id, valid value, non-empty quote.
        try:
            finding = Finding.model_validate(item)
        except ValidationError:
            logger.warning("Dropping invalid finding: %r", item)
            continue

        # 2. Grounding check: the quote must really appear in the message.
        quote = _normalise(finding.quote)
        if not quote or quote not in normalised_message:
            logger.warning("Dropping ungrounded finding: %r", item)
            continue

        # 3. Duplicates: keep the most cautious value for each sign.
        current = best.get(finding.sign_id)
        if current is None or CAUTION_RANK[finding.value] > CAUTION_RANK[current.value]:
            best[finding.sign_id] = finding

    return list(best.values())


def run_extraction(message: str, llm: LLMFunc) -> ExtractResponse:
    """The full pipeline. `llm` is passed in"""
    raw = llm(message)
    findings = clean_findings(raw, message)
    escalate, reason = decide_escalation(findings)
    return ExtractResponse(findings=findings, escalate=escalate, reason=reason)
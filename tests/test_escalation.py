"""Escalation truth table. Every urgent sign x every value."""

import pytest

from app.escalation import ESCALATION_SIGNS, decide_escalation
from app.schemas import Finding, SignId, Value


def make(sign: SignId, value: Value) -> Finding:
    """Small helper to build a finding; the quote is irrelevant to escalation."""
    return Finding(sign_id=sign, value=value, quote="x")


@pytest.mark.parametrize("sign", list(ESCALATION_SIGNS))
@pytest.mark.parametrize("value", [Value.YES, Value.NOT_SURE])
def test_urgent_sign_yes_or_not_sure_escalates(sign, value):
    escalate, reason = decide_escalation([make(sign, value)])
    assert escalate is True
    assert sign.value in reason


@pytest.mark.parametrize("sign", list(ESCALATION_SIGNS))
def test_urgent_sign_explicitly_absent_does_not_escalate(sign):
    escalate, _ = decide_escalation([make(sign, Value.NO)])
    assert escalate is False


@pytest.mark.parametrize(
    "sign", [s for s in SignId if s not in ESCALATION_SIGNS]
)
def test_non_urgent_signs_never_escalate(sign):
    escalate, _ = decide_escalation([make(sign, Value.YES)])
    assert escalate is False


def test_no_findings_does_not_escalate():
    escalate, reason = decide_escalation([])
    assert escalate is False
    assert reason


def test_reason_lists_every_trigger():
    findings = [
        make(SignId.FROTH_NOSE_MOUTH, Value.YES),
        make(SignId.UNABLE_TO_STAND, Value.NOT_SURE),
        make(SignId.TICKS, Value.YES),  # not urgent, so not in the reason
    ]
    escalate, reason = decide_escalation(findings)
    assert escalate is True
    assert "froth_nose_mouth (yes)" in reason
    assert "unable_to_stand (not_sure)" in reason
    assert "ticks" not in reason
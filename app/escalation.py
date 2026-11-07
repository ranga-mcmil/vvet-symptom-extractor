"""Escalation decision. Deliberately plain code, with no LLM involved.

This is a safety rule, so it must be deterministic, easy to read, and easy to
test. Changing which signs are urgent is a one-line change below.
"""

from app.schemas import Finding, SignId, Value

# Signs that make a case urgent...
ESCALATION_SIGNS = frozenset(
    {SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND, SignId.CLOUDY_EYE}
)

# ...when the farmer says "yes" OR sounds unsure. We err on the side of caution:
# a false alarm is cheap, a missed emergency is not.
ESCALATION_VALUES = frozenset({Value.YES, Value.NOT_SURE})


def decide_escalation(findings: list[Finding]) -> tuple[bool, str]:
    """Return (escalate, reason) for a list of validated findings."""
    triggers = [
        f
        for f in findings
        if f.sign_id in ESCALATION_SIGNS and f.value in ESCALATION_VALUES
    ]

    if not triggers:
        return False, "None of the urgent signs were reported."

    # The reason is built from the findings themselves, not written by an LLM,
    # so it can never contradict the decision.
    described = ", ".join(f"{f.sign_id.value} ({f.value.value})" for f in triggers)
    return True, f"Urgent sign(s) reported: {described}"
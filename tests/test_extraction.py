"""Tests for validating model output. The LLM is a plain fake function."""

import pytest

from app.errors import LLMError
from app.extraction import clean_findings, run_extraction
from app.schemas import SignId, Value

MESSAGE = "There's foam coming from her nose and she can't get up."


def raw(*findings):
    """Build the dict shape the model is supposed to return."""
    return {"findings": list(findings)}


def f(sign, value, quote):
    return {"sign_id": sign, "value": value, "quote": quote}


def test_valid_findings_pass_through():
    result = clean_findings(
        raw(
            f("froth_nose_mouth", "yes", "foam coming from her nose"),
            f("unable_to_stand", "yes", "can't get up"),
        ),
        MESSAGE,
    )
    assert [x.sign_id for x in result] == [
        SignId.FROTH_NOSE_MOUTH,
        SignId.UNABLE_TO_STAND,
    ]


def test_unknown_sign_id_is_dropped():
    result = clean_findings(raw(f("limping", "yes", "can't get up")), MESSAGE)
    assert result == []


@pytest.mark.parametrize("bad_value", ["maybe", "YES", "", None])
def test_invalid_value_is_dropped(bad_value):
    result = clean_findings(raw(f("unable_to_stand", bad_value, "can't get up")), MESSAGE)
    assert result == []


def test_hallucinated_quote_is_dropped():
    # The model claims ticks, but the message never mentions them.
    result = clean_findings(raw(f("ticks", "yes", "lots of ticks")), MESSAGE)
    assert result == []


def test_empty_or_whitespace_quote_is_dropped():
    result = clean_findings(raw(f("ticks", "yes", "   ")), MESSAGE)
    assert result == []


def test_quote_matching_ignores_case_and_curly_apostrophes():
    message = "She CAN\u2019T get up"  # curly apostrophe, as WhatsApp sends
    result = clean_findings(raw(f("unable_to_stand", "yes", "can't get up")), message)
    assert len(result) == 1


def test_bad_items_do_not_ruin_good_ones():
    result = clean_findings(
        raw(
            "not even a dict",
            f("limping", "yes", "can't get up"),
            f("unable_to_stand", "yes", "can't get up"),
        ),
        MESSAGE,
    )
    assert [x.sign_id for x in result] == [SignId.UNABLE_TO_STAND]


def test_duplicate_sign_keeps_most_cautious_value():
    result = clean_findings(
        raw(
            f("unable_to_stand", "no", "can't get up"),
            f("unable_to_stand", "yes", "can't get up"),
        ),
        MESSAGE,
    )
    assert len(result) == 1
    assert result[0].value == Value.YES


@pytest.mark.parametrize("garbage", [None, [], "text", {}, {"findings": "nope"}])
def test_unusable_output_raises(garbage):
    with pytest.raises(LLMError):
        clean_findings(garbage, MESSAGE)


def test_empty_findings_list_is_valid():
    assert clean_findings({"findings": []}, MESSAGE) == []


def test_pipeline_end_to_end_with_fake_llm():
    def fake_llm(message):
        return raw(f("unable_to_stand", "yes", "can't get up"))

    response = run_extraction(MESSAGE, fake_llm)
    assert response.escalate is True
    assert "unable_to_stand" in response.reason


def test_hallucinated_urgent_sign_cannot_trigger_escalation():
    """The key safety property: a made-up sign must not cause a false escalation."""

    def fake_llm(message):
        return raw(f("unable_to_stand", "yes", "she collapsed"))  # not in message

    response = run_extraction("The calf has ticks on the ears.", fake_llm)
    assert response.escalate is False
    assert response.findings == []
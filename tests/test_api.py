"""End-to-end API tests. The LLM is always a fake, so no network or API key."""

import pytest

from app.errors import LLMError


def f(sign, value, quote):
    return {"sign_id": sign, "value": value, "quote": quote}


# (message, what the fake LLM returns, expected {sign: value}, expected escalate)
SAMPLES = [
    pytest.param(
        "My cow has not been eating since yesterday and she is breathing very fast.",
        [
            f("reduced_appetite", "yes", "has not been eating"),
            f("rapid_breathing", "yes", "breathing very fast"),
        ],
        {"reduced_appetite": "yes", "rapid_breathing": "yes"},
        False,
        id="appetite-and-breathing",
    ),
    pytest.param(
        "There's foam coming from her nose and she can't get up.",
        [
            f("froth_nose_mouth", "yes", "foam coming from her nose"),
            f("unable_to_stand", "yes", "can't get up"),
        ],
        {"froth_nose_mouth": "yes", "unable_to_stand": "yes"},
        True,
        id="foam-and-cannot-stand",
    ),
    pytest.param(
        "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear.",
        [
            f("cloudy_eye", "not_sure", "the eyes look a bit cloudy but I'm not sure"),
            f("swelling", "yes", "mbandabvu under the ear"),
        ],
        {"cloudy_eye": "not_sure", "swelling": "yes"},
        True,
        id="unsure-cloudy-eye-and-swelling",
    ),
    pytest.param(
        "The calf is fine, just lots of ticks on the ears.",
        [f("ticks", "yes", "lots of ticks")],
        {"ticks": "yes"},
        False,
        id="calf-fine-ticks-only",
    ),
    pytest.param(
        "Mombe yangu haisi kudya, and her urine is dark like tea.",
        [
            f("reduced_appetite", "yes", "haisi kudya"),
            f("red_urine", "yes", "urine is dark like tea"),
        ],
        {"reduced_appetite": "yes", "red_urine": "yes"},
        False,
        id="shona-mixed",
    ),
]


@pytest.mark.parametrize("message, llm_findings, expected, escalate", SAMPLES)
def test_sample_messages(make_client, message, llm_findings, expected, escalate):
    client = make_client(lambda _msg: {"findings": llm_findings})

    response = client.post("/extract", json={"message": message})

    assert response.status_code == 200
    body = response.json()
    assert {x["sign_id"]: x["value"] for x in body["findings"]} == expected
    assert body["escalate"] is escalate
    assert body["reason"]


def test_invalid_and_hallucinated_findings_are_filtered_out(make_client):
    fake = lambda _msg: {  # noqa: E731
        "findings": [
            f("ticks", "yes", "lots of ticks"),  # valid
            f("limping", "yes", "lots of ticks"),  # unknown sign
            f("unable_to_stand", "yes", "she collapsed"),  # quote not in message
        ]
    }
    client = make_client(fake)

    body = client.post(
        "/extract", json={"message": "The calf has lots of ticks."}
    ).json()

    assert [x["sign_id"] for x in body["findings"]] == ["ticks"]
    assert body["escalate"] is False  # the made-up urgent sign did not leak through


def test_llm_failure_returns_503_not_a_false_all_clear(make_client):
    def broken(_msg):
        raise LLMError("provider down")

    client = make_client(broken)

    response = client.post("/extract", json={"message": "She can't get up."})

    assert response.status_code == 503
    assert "escalate" not in response.json()


def test_garbage_model_output_returns_503(make_client):
    client = make_client(lambda _msg: {"unexpected": "shape"})
    response = client.post("/extract", json={"message": "She can't get up."})
    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": ""}, {"message": "   "}, {"message": "x" * 2001}, {"message": 123}],
)
def test_bad_requests_are_rejected_with_422(make_client, payload):
    client = make_client(lambda _msg: {"findings": []})
    assert client.post("/extract", json=payload).status_code == 422
# V-Vet Symptom Extraction Service

A small FastAPI service that reads a farmer's description of a sick animal and returns structured findings from a fixed list of nine signs, plus an escalation decision.

```
POST /extract
{"message": "There's foam coming from her nose and she can't get up."}

200 OK
{
  "findings": [
    {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
    {"sign_id": "unable_to_stand",  "value": "yes", "quote": "can't get up"}
  ],
  "escalate": true,
  "reason": "Urgent sign(s) reported: froth_nose_mouth (yes), unable_to_stand (yes)"
}
```

The core idea: **the LLM reads, plain code decides.** The model suggests findings, every suggestion is checked by code, and the escalation decision never touches the model.

---

## Run it

Requires Python 3.11+ and a free Groq API key (https://console.groq.com, no credit card needed).

```bash
git clone <this-repo>
cd vvet-symptom-extractor

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env                   # then paste your GROQ_API_KEY into .env
uvicorn app.main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Or with curl:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "The calf is fine, just lots of ticks on the ears."}'
```

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq key. Never committed; `.env` is git-ignored. | (required) |
| `GROQ_MODEL` | Which Groq model to use | `openai/gpt-oss-20b` |

### Run the tests

```bash
pytest
```

The tests need **no network and no API key**. The LLM is replaced by a plain fake function, and the Groq client is replaced by a fake object in the provider tests.

---

## How it works

```
request ─► main.py ─► extraction.run_extraction
                          │
                          ├─ 1. llm.py             ask Groq            (untrusted)
                          ├─ 2. extraction.py      validate and clean  (code)
                          └─ 3. escalation.py      decide escalate     (code)
```

| File | Responsibility |
|---|---|
| `app/main.py` | Thin web layer: parse request, call pipeline, map errors to HTTP |
| `app/schemas.py` | Pydantic models and the fixed `SignId` / `Value` enums |
| `app/llm.py` | The only module that knows about Groq: prompt, API call, JSON parsing |
| `app/extraction.py` | Validates model output and runs the pipeline |
| `app/escalation.py` | Pure function: findings → `(escalate, reason)` |
| `app/errors.py` | One exception, `LLMError` |

---

## Decisions and why

**1. Escalation is plain code, never the LLM.**
It's a safety decision, so it must be deterministic, auditable and testable. `escalate` and `reason` are computed from the validated findings, so the reason can never contradict the decision.

**2. The model's output is treated as untrusted and checked twice.**
JSON mode is requested from the provider, but it only improves the odds of valid JSON. Afterwards every finding goes through Pydantic, where the sign ID and value are enums, so an unknown sign or a value like `"maybe"` can't pass. Invalid findings are dropped individually, so one bad item doesn't lose the good ones. If the whole response is unusable (not an object, no `findings` list), the request fails.

**3. Quote grounding.**
Every finding must include a `quote`, and that quote must actually appear in the farmer's message (compared case-insensitively, with whitespace and curly WhatsApp apostrophes normalised). A finding whose quote isn't in the message is dropped. This turns "only report signs the message mentions" from a request in the prompt into something the code enforces. It also means a made-up urgent sign cannot cause a false escalation. There's a test for exactly that.

**4. Duplicates keep the most cautious value** (`yes` > `not_sure` > `no`), so the model contradicting itself can't lead to under-escalation.

**5. `not_sure` escalates for the three urgent signs.**
A false alarm costs a vet's attention. A missed emergency costs an animal. When a farmer is unsure about cloudy eyes, that's exactly when a vet should look.

**6. A failing LLM returns HTTP 503, never `escalate: false`.**
If the model times out, errors, or returns garbage, the API says it's unavailable. A silent "no escalation" on a sick animal would be the worst possible failure. The real cause is logged; the client gets a generic message.

**7. Only mentioned signs are returned.**
If breathing isn't mentioned, there's no `rapid_breathing: no`. Not mentioned is not the same as absent. `no` is reserved for when the farmer explicitly says a sign is absent.

**8. The LLM is isolated behind one function.**
`run_extraction(message, llm)` takes the LLM as a parameter, and FastAPI injects it with `Depends`. That's how tests substitute a fake with no mocking of globals, and switching provider means rewriting only `llm.py`.

**9. Prompt design.**
Temperature 0, the sign definitions, explicit rules (only mentioned signs, exact quotes, when to use `not_sure` and `no`), four few-shot examples (a `not_sure` case, an explicit `no`, a two-sign message and an empty result), and the farmer's text wrapped in `<message>` tags with an instruction to treat it as data, as basic prompt-injection resistance. The code-side checks are the real defence, since prompts can be ignored.

**10. Model choice: `openai/gpt-oss-20b` on Groq's free tier.**
My first choice, `llama-3.3-70b-versatile`, returned a 404 (`model_not_found`) because Groq lists it as Enterprise-only. That was fixed with a one-line config change, which is why the model is an environment variable and not hardcoded. The client also sets a 15-second timeout and relies on the SDK's built-in retries (2) for transient errors.

**11. Input limits.** Messages are trimmed, then must be 1 to 2000 characters. Anything else gets an automatic `422` from FastAPI before any LLM call is made.

**12. Plain `def` endpoint, not `async def`.**
The Groq client is synchronous. FastAPI runs sync endpoints in a thread pool, so a slow model call doesn't block the server.

---

## Observations from running the sample messages

I ran the five sample messages against the real model (`openai/gpt-oss-20b`), printing both the raw model output and the final validated response.

| Message | Findings returned | Escalate |
|---|---|---|
| "My cow has not been eating since yesterday and she is breathing very fast." | `reduced_appetite: yes`, `rapid_breathing: yes` | false |
| "There's foam coming from her nose and she can't get up." | `froth_nose_mouth: yes`, `unable_to_stand: yes` | **true** |
| "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear." | `cloudy_eye: not_sure`, `swelling: yes` | **true** |
| "The calf is fine, just lots of ticks on the ears." | `ticks: yes` | false |
| "Mombe yangu haisi kudya, and her urine is dark like tea." | `reduced_appetite: yes`, `red_urine: yes` | false |

What I noticed:

- **All five were correct**, including the `not_sure` case (escalated because of `cloudy_eye`) and the mixed Shona/English message.
- **Nothing was filtered.** The raw model output and the final response were identical for every message, so the quote-grounding check and the sign validation had nothing to remove in these runs. They are covered by the unit tests instead, using fake model output.
- **"The calf is fine" produced no finding**, as intended. The model reported only the ticks.
- **Quotes are valid but not always minimal.** For example, `"There's foam coming from her nose"`, `"She has mbandabvu under the ear."` (including the full stop) and `"just lots of ticks on the ears"`. The prompt asks for the shortest supporting text, so this is a prompt-tuning opportunity. Every quote did appear in the message, which is all the grounding check requires.
- **The cloudy-eye quote omits "but I'm not sure".** The value was still correctly `not_sure`, but it shows the quote doesn't always carry the uncertainty with it.
- **Caveat on the Shona result:** the prompt's glossary includes `mbandabvu` and `haisi kudya`, which are the terms from these sample messages, so this run shows the glossary working and not how the model copes with Shona it hasn't been given hints for.
- **This is five messages.** It's a smoke test, not an accuracy measurement. See the plan below for how I'd measure properly.

---

## Limitations (what this does not do)

- **Omissions go unnoticed.** Quote grounding catches findings the model *invents*, but not signs it *forgets*. If the farmer says "can't get up" and the model returns nothing, the code can't tell. This is the biggest risk, and it's why recall on the urgent signs is my headline accuracy metric below.
- **Shona support is thin.** The prompt glossary contains only terms that appear in the brief itself (`mbandabvu`, `haisi kudya`, `mombe`). I deliberately did not add Shona words I couldn't verify. The model is probably weaker in Shona than English, and I haven't measured by how much.
- **Single message only.** No conversation history or follow-ups.
- **No authentication or rate limiting** on the endpoint.
- **Free-tier limits.** Groq's free tier is rate-limited, so this isn't sized for production traffic.

---

## How I would measure accuracy

The unit tests prove the *code* is right. Measuring the *model* needs a labelled dataset.

1. **Build a labelled set of 100+ messages.** English, Shona, mixed, vague, negated ("she is *not* breathing fast"), and multi-sign. Real farmer messages if V-Vet can share them. Each is annotated with the correct `(sign_id, value)` pairs, ideally by two people (a vet or para-vet, and a Shona-fluent reviewer), with disagreements resolved and agreement measured (Cohen's kappa). Labelling quality caps how good the measurement can be.
2. **Headline metric: recall on `froth_nose_mouth`, `unable_to_stand` and `cloudy_eye`**, counting `yes` and `not_sure` as positive. A miss on these is a missed escalation, so this matters more than anything else. I'd track escalation-level recall and the false-alarm rate directly, and set a minimum recall target before any release.
3. **Per-sign precision and recall** for all nine signs.
4. **Value accuracy.** A yes / no / not_sure confusion matrix, since getting the *value* wrong (for example `not_sure` read as `no`) is a subtle failure.
5. **Quote-grounding drop rate.** How often the raw model output has to be filtered. A rising rate signals hallucination or a silent model change.
6. **Slice by language.** Report English, Shona and mixed separately so a strong English score can't hide weak Shona.
7. **Regression runs.** Re-run the whole set on every prompt or model change, and repeat runs to check stability.

---

## With more time

1. **Build the labelled set and an evaluation script**, then improve the prompt against it. This is the first thing I'd do.
2. **A deterministic keyword safety net** for the three urgent signs (including Shona terms reviewed by a fluent speaker) as a backstop against model omissions, the main gap above.
3. **Resilience:** application-level retries with backoff, a circuit breaker, and a fallback provider.
4. **Observability:** structured logs with request IDs, plus dashboards for latency, 503 rate and the grounding drop rate, to catch drift.
5. **Cost and speed:** cache repeated messages; consider schema-constrained output where the provider supports it.
6. **Conversation handling:** farmers often add detail across several messages.

---

## How I used AI tools

I used **Claude (Anthropic)** as an AI coding assistant on this project:

- **Design.** I discussed the architecture with it before writing code: where the LLM boundary sits, how to validate model output, and how to keep escalation in plain code.
- **Implementation.** It drafted the code and tests, which I then typed in, ran and committed step by step.
- **Debugging.** When the first Groq model returned a 404, I checked Groq's model list, found it was Enterprise-only, and switched to `openai/gpt-oss-20b`.
- **Documentation.** It helped draft this README, and I filled in the observations from my own runs.

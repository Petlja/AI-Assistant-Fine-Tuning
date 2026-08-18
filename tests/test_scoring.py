"""Tests for the pointwise scorer.

No network: every test drives `score_answer` with a stub client. The point is
to pin the contract AI-Assistant-LLM-Compare calibrates against, so a change
that silently alters what the teacher model is asked, or what comes back,
fails here rather than halfway through a paid calibration run.
"""

from __future__ import annotations

import sys
import types

import pytest

# scoring.py must stay importable without the training stack; blocking these
# here means the test fails if someone gives it a heavyweight import.
for _blocked in ("plct_ai_data_unifier", "torch", "vllm", "transformers", "trl", "peft"):
    sys.modules.setdefault(_blocked, None)

from ai_assistant_fine_tuning.scoring import (  # noqa: E402
    DEFAULT_SCALE_MAX,
    DEFAULT_THRESHOLD,
    EVALUATION_PROMPT,
    SCORER_VARIANTS,
    EvaluationResult,
    Score,
    ScorerConfig,
    build_scoring_messages,
    get_scorer,
    score_answer,
    transform_evaluate_batch,
)


class _StubClient:
    """Minimal stand-in for `OpenAI`, recording the call it was given."""

    def __init__(self, parsed=None, content="{}", refusal=None):
        self._parsed = parsed
        self._content = content
        self._refusal = refusal
        self.calls: list[dict] = []

        message = types.SimpleNamespace(
            parsed=parsed, content=content, refusal=refusal
        )
        completion = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )

        def parse(**kwargs):
            self.calls.append(kwargs)
            return completion

        self.beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(parse=parse))
        )


def _score(**kwargs):
    """Run score_answer against a stub returning a valid result."""
    parsed = EvaluationResult(
        score=kwargs.pop("score", 88),
        improved_answer=kwargs.pop("improved_answer", "better answer"),
    )
    client = _StubClient(parsed=parsed, content=kwargs.pop("content", '{"score": 88}'))
    result = score_answer(
        client,
        system_message=kwargs.pop("system_message", "SYSMSG"),
        question=kwargs.pop("question", "QUESTION"),
        answer=kwargs.pop("answer", "ANSWER"),
        config=kwargs.pop("config", get_scorer()),
    )
    return client, result


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_score_answer_returns_score_with_provenance():
    _, result = _score(score=88, improved_answer="better answer")

    assert isinstance(result, Score)
    assert result.score == 88
    assert result.scale_max == 100
    assert result.improved_answer == "better answer"
    assert result.scorer == "v1-baseline"


def test_score_answer_sends_one_answer_and_the_config_settings():
    client, _ = _score()
    (call,) = client.calls

    assert call["model"] == "gpt-5.4-mini"
    assert call["temperature"] == 0.3
    assert call["response_format"] is EvaluationResult

    system, user = call["messages"]
    assert system["role"] == "system"
    assert user["role"] == "user"
    # The scorer is pointwise: the prompt carries exactly one answer.
    assert "ANSWER" in user["content"]
    assert "QUESTION" in user["content"]
    assert "SYSMSG" in user["content"]


def test_prompt_asks_for_the_full_1_100_range():
    # The scale and its anchors are what the whole calibration rests on.
    assert "1 to 100" in EVALUATION_PROMPT
    assert "86-100" in EVALUATION_PROMPT
    assert "90-100" in EVALUATION_PROMPT
    assert "Use the full range" in EVALUATION_PROMPT


def test_improved_answer_is_always_requested():
    # Locked decision: improved_answer is generated in calibration runs too.
    assert "improved_answer" in EvaluationResult.model_fields
    assert "improved_answer" in EVALUATION_PROMPT


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


def test_refusal_raises_rather_than_returning_none():
    client = _StubClient(parsed=None, content="", refusal="I cannot help with that")
    with pytest.raises(ValueError, match="refusal"):
        score_answer(
            client,
            system_message="s",
            question="q",
            answer="a",
            config=get_scorer(),
        )


@pytest.mark.parametrize("bad", [0, 101, -5])
def test_wire_model_rejects_out_of_range_scores(bad):
    with pytest.raises(Exception):
        EvaluationResult(score=bad, improved_answer="x")


def test_score_above_config_scale_max_is_rejected():
    # A variant on a narrower scale must not silently accept a 100.
    narrow = ScorerConfig(name="narrow", scale_max=50)
    parsed = EvaluationResult(score=90, improved_answer="x")
    client = _StubClient(parsed=parsed)
    with pytest.raises(ValueError, match="outside 1..50"):
        score_answer(
            client, system_message="s", question="q", answer="a", config=narrow
        )


# --------------------------------------------------------------------------
# Batch path
# --------------------------------------------------------------------------


def test_transform_evaluate_batch_accepts_the_new_scale():
    result = transform_evaluate_batch({"score": 95, "improved_answer": "x"})
    assert result.score == 95


@pytest.mark.parametrize(
    "payload",
    [
        {"score": 0, "improved_answer": "x"},
        {"score": 101, "improved_answer": "x"},
        {"score": "88", "improved_answer": "x"},
        {"score": 88, "improved_answer": ""},
        {"score": 88},
    ],
)
def test_transform_evaluate_batch_rejects_bad_payloads(payload):
    with pytest.raises(ValueError):
        transform_evaluate_batch(payload)


# --------------------------------------------------------------------------
# Variant registry
# --------------------------------------------------------------------------


def test_baseline_variant_matches_the_shipped_defaults():
    config = get_scorer("v1-baseline")
    assert config.scale_max == DEFAULT_SCALE_MAX
    assert config.prompt_template == EVALUATION_PROMPT


def test_model_override_does_not_mutate_the_registry():
    overridden = get_scorer("v1-baseline", model="gpt-4o")
    assert overridden.model == "gpt-4o"
    # Calibration results are recorded against a variant name; if an override
    # leaked into the registry those records would become meaningless.
    assert SCORER_VARIANTS["v1-baseline"].model == "gpt-5.4-mini"


def test_unknown_variant_names_the_known_ones():
    with pytest.raises(ValueError, match="v1-baseline"):
        get_scorer("does-not-exist")


def test_threshold_default_sits_on_the_1_100_scale():
    assert 1 <= DEFAULT_THRESHOLD <= DEFAULT_SCALE_MAX
    assert DEFAULT_THRESHOLD == 70


def test_build_scoring_messages_is_what_score_answer_sends():
    # The batch path builds requests through this helper; if it drifts from
    # what the sync path sends, the two paths score differently.
    config = get_scorer()
    built = build_scoring_messages(
        system_message="SYSMSG", question="QUESTION", answer="ANSWER", config=config
    )
    client, _ = _score()
    assert client.calls[0]["messages"] == built

"""LLM-as-judge for RAG generation quality.

Implements two judges scored 0-5 (mapped to 0-1):

- ``Faithfulness``: the answer's claims are supported by the provided
  source documents (no hallucinations / no contradictions).
- ``Answer relevance``: the answer addresses the user question
  (not off-topic, not a meta-comment).

Design choices:
- Pluggable model via env (``JUDGE_MODEL``) defaulting to the generator
  model. Self-preference bias is logged as a warning when judge ==
  generator.
- Strict structured output (JSON schema with score 0-5 + 1-line
  justification). Failures fall back to None and are reported in the
  aggregate as "judge_unparseable_rate".
- Single LLM call per (judge, query). Parallelization is the caller's
  responsibility.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger


_JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "reasoning": {"type": "string", "maxLength": 280},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


FAITHFULNESS_SYSTEM_PROMPT = """\
You are a strict evaluator of factual grounding for Brazilian aviation
regulation answers.

Given a USER QUESTION, an ANSWER produced by an assistant, and the SOURCE
DOCUMENTS the assistant was supposed to use, decide how faithfully the
answer is grounded in the sources.

SCORING RUBRIC (0-5):
  5 - Every factual claim in the answer is directly supported by the
      sources. No hallucinations. Citations (when present) point to the
      correct source.
  4 - Almost all claims are supported; at most one minor unsupported
      detail that doesn't affect the answer's correctness.
  3 - Most claims are supported but at least one unsupported claim
      changes the meaning.
  2 - Several unsupported claims OR clearly fabricated details.
  1 - Mostly unsupported / hallucinated.
  0 - Completely unsupported, contradicts the sources, or invented.

If the answer is "Não encontrei essa informação nos documentos
disponíveis." AND the sources truly don't address the question, score 5
(faithful refusal). If the sources DO address the question but the model
refused, score 1.

Reply ONLY with valid JSON: {"score": <int 0-5>, "reasoning": "<one line>"}.
"""


RELEVANCE_SYSTEM_PROMPT = """\
You are a strict evaluator of answer relevance for Brazilian aviation
regulation queries.

Given a USER QUESTION and the ANSWER, decide whether the answer
ADDRESSES the question (regardless of whether it is correct).

SCORING RUBRIC (0-5):
  5 - Directly and completely addresses every part of the question.
  4 - Addresses the main intent; secondary aspects partially covered.
  3 - Addresses the topic but partial / generic.
  2 - Tangentially related, mostly off-topic.
  1 - Off-topic with token overlap only.
  0 - Completely off-topic OR meta-commentary about not being able to
      answer when the user asked a clear factual question.

A well-grounded refusal "Não encontrei essa informação nos documentos
disponíveis." for a clear factual question scores 2 (acknowledged
the question but didn't answer it).

Reply ONLY with valid JSON: {"score": <int 0-5>, "reasoning": "<one line>"}.
"""


@dataclass
class JudgeVerdict:
    """Result of a single judgement call."""
    score: Optional[float]
    raw_score: Optional[int]
    reasoning: str
    parse_error: Optional[str] = None


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the LLM's JSON output into a JudgeVerdict."""
    raw_stripped = raw.strip()

    if raw_stripped.startswith("```"):
        match = re.search(r"\{.*\}", raw_stripped, re.DOTALL)
        if match:
            raw_stripped = match.group(0)

    try:
        data = json.loads(raw_stripped)
    except json.JSONDecodeError as e:
        return JudgeVerdict(
            score=None, raw_score=None,
            reasoning="", parse_error=f"json_decode: {e}",
        )

    score = data.get("score")
    if not isinstance(score, int) or score < 0 or score > 5:
        return JudgeVerdict(
            score=None, raw_score=None,
            reasoning=str(data.get("reasoning", "")),
            parse_error=f"invalid_score: {score!r}",
        )

    return JudgeVerdict(
        score=score / 5.0,
        raw_score=score,
        reasoning=str(data.get("reasoning", "")),
    )


def _format_sources(sources: List[dict], char_budget: int = 4000) -> str:
    """Render sources for the judge prompt with a soft character cap."""
    parts: List[str] = []
    used = 0
    for i, s in enumerate(sources, 1):
        reg_id = s.get("regulation_id") or s.get("doc_id") or "?"
        text = (s.get("text") or "").strip()
        block = f"[Source {i}] {reg_id}\n{text}"
        if used + len(block) > char_budget and parts:
            parts.append(f"\n[... {len(sources) - i + 1} more sources truncated ...]")
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


class LLMJudge:
    """Wraps an LLM as a structured judge for RAG generation."""

    def __init__(self, llm, model_name: str, generator_model: Optional[str] = None):
        self._llm = llm
        self.model_name = model_name
        if generator_model and generator_model == model_name:
            logger.warning(
                "LLMJudge: judge model ({m}) == generator model — "
                "self-preference bias likely. Consider JUDGE_MODEL=<other>.",
                m=model_name,
            )

    def _judge_one(self, system: str, user: str) -> JudgeVerdict:
        try:
            raw = self._llm.generate(
                prompt=user,
                system_prompt=system,
                temperature=0.0,
                max_tokens=256,
                format=_JUDGE_OUTPUT_SCHEMA,
                extra_options={
                    # Critical: Gemma 4 ships with 256K default ctx → 80GB+ KV
                    # cache, evicting other models from VRAM. The judge prompt
                    # is short (system + question + answer + truncated sources
                    # ≈ 1.5K tokens), so 4K is plenty and keeps the model
                    # resident on a single L40S alongside the generator and
                    # cross-encoder.
                    "num_ctx": 4096,
                },
            )
        except Exception as e:
            return JudgeVerdict(
                score=None, raw_score=None, reasoning="",
                parse_error=f"llm_error: {e}",
            )
        return _parse_verdict(raw or "")

    def judge_faithfulness(
        self,
        question: str,
        answer: str,
        sources: List[dict],
    ) -> JudgeVerdict:
        user = (
            f"USER QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"SOURCE DOCUMENTS:\n{_format_sources(sources)}\n"
        )
        return self._judge_one(FAITHFULNESS_SYSTEM_PROMPT, user)

    def judge_relevance(self, question: str, answer: str) -> JudgeVerdict:
        user = f"USER QUESTION:\n{question}\n\nANSWER:\n{answer}\n"
        return self._judge_one(RELEVANCE_SYSTEM_PROMPT, user)

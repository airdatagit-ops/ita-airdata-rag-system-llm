"""Response generator module.

Produces the final user-facing answer from evaluated documents, the
original query, and optional conversation history.
"""

from __future__ import annotations

from typing import Dict, Generator, List, Optional

from loguru import logger

from config import config
from models.llm import LlamaModel
from search.shared.exceptions import GeneratorError
from search.shared.schemas import EvaluatedDocument
from search.shared.timeouts import with_timeout
from search.generator.prompts import (
    GENERATOR_GROUNDED_SYSTEM_PROMPT,
    GENERATOR_UNGROUNDED_SYSTEM_PROMPT,
    GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT,
    GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT,
    build_generator_context,
    build_generator_prompt,
)


class ResponseGenerator:
    """Generates the final answer using an LLM with evaluated context.

    Parameters
    ----------
    llm:
        LLM instance for generation.  When *None*, a dedicated instance
        is created using ``GENERATOR_MODEL``.
    grounded_only:
        Default groundedness mode.  ``True`` means answers must come
        exclusively from the provided documents.
    max_tokens:
        Maximum response tokens.
    timeout:
        Hard timeout (seconds) for the LLM generation call.
    """

    def __init__(
        self,
        llm: Optional[LlamaModel] = None,
        grounded_only: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        model_name = config.GENERATOR_MODEL or config.OLLAMA_MODEL
        self.llm = llm or LlamaModel(model_name=model_name)
        self.grounded_only = (
            grounded_only if grounded_only is not None
            else config.GENERATOR_GROUNDED_ONLY
        )
        self.max_tokens = max_tokens or config.GENERATOR_MAX_RESPONSE_TOKENS
        self.timeout = timeout or config.GENERATOR_TIMEOUT
        logger.info(
            f"ResponseGenerator initialized (model={model_name}, "
            f"grounded={self.grounded_only}, timeout={self.timeout}s)"
        )

    def generate(
        self,
        evaluated_docs: List[EvaluatedDocument],
        original_query: str,
        *,
        rewritten_queries: List[str] | None = None,
        history: List[Dict[str, str]] | None = None,
        grounded_only: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Generate the final response.

        Returns a string (non-streaming) or a generator of chunks
        (streaming).
        """
        effective_grounded = (
            grounded_only if grounded_only is not None
            else self.grounded_only
        )
        effective_max_tokens = max_tokens or self.max_tokens

        documents = [ed.document for ed in evaluated_docs]
        scores = [ed.relevance_score for ed in evaluated_docs]

        context = build_generator_context(documents, scores)
        prompt = build_generator_prompt(
            original_query, context, history=history,
        )
        system_prompt = self._select_system_prompt(
            grounded=effective_grounded, has_history=bool(history),
        )

        if stream:
            return self._generate_stream(
                prompt, system_prompt, temperature, effective_max_tokens,
            )

        return self._generate_sync(
            prompt, system_prompt, temperature, effective_max_tokens,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float],
        max_tokens: int,
    ) -> str:
        fallback_msg = (
            "Não foi possível gerar a resposta no tempo disponível. "
            "Tente novamente ou reformule sua pergunta."
        )

        def _call():
            return self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        answer, timed_out = with_timeout(
            _call,
            timeout_seconds=self.timeout,
            fallback_value=fallback_msg,
            stage_name="Generator",
        )

        if timed_out:
            return fallback_msg

        return answer

    def _generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float],
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Streaming generation — yields chunks then appends references."""
        try:
            stream = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                yield chunk

        except Exception as exc:
            logger.error(f"Streaming generation error: {exc}")
            raise GeneratorError(f"Streaming generation failed: {exc}") from exc

    @staticmethod
    def _select_system_prompt(*, grounded: bool, has_history: bool) -> str:
        if has_history:
            return (
                GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT
                if grounded
                else GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT
            )
        return (
            GENERATOR_GROUNDED_SYSTEM_PROMPT
            if grounded
            else GENERATOR_UNGROUNDED_SYSTEM_PROMPT
        )

"""Diagnostic script: test Rewriter, Searcher and Evaluator independently.

Usage:  python test_pipeline_stages.py [query]
Default query: "regras para drones"

Prints detailed output for each stage so you can pinpoint failures.
Respects INFERENCE_MODE (local/remote/cpu) from .env.
"""

import sys
import time
import json

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level:<7} | {message}")

from config import config
from models.gpu_client import create_llm, create_evaluator

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "regras para drones"

print("=" * 70)
print(f"QUERY: {query!r}")
print(f"INFERENCE_MODE: {config.INFERENCE_MODE}")
if config.INFERENCE_MODE == "remote":
    print(f"GPU_SERVER_URL: {config.GPU_SERVER_URL}")
print("=" * 70)


# ── 1. REWRITER ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STAGE 1: REWRITER")
print("─" * 70)

from search.rewriter.prompts import build_system_prompt, build_user_prompt
from search.rewriter.rewriter import _RewriterOutput, QueryRewriter

system_prompt = build_system_prompt(config.REWRITER_MAX_QUERIES)
user_prompt = build_user_prompt(query)

print(f"\nModel: {config.REWRITER_MODEL}")
print(f"Temperature: {config.REWRITER_TEMPERATURE}")
print(f"System prompt ({len(system_prompt)} chars):")
print(f"  {system_prompt[:200]}...")
print(f"User prompt: {user_prompt!r}")

rewriter_llm = create_llm(model_name=config.REWRITER_MODEL)
output_schema = _RewriterOutput.model_json_schema()

t0 = time.time()
raw_output = rewriter_llm.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=config.REWRITER_TEMPERATURE,
    format=output_schema,
)
rewriter_ms = int((time.time() - t0) * 1000)

print(f"\nRaw LLM output ({rewriter_ms}ms):")
print(f"  {raw_output!r}")

rewriter = QueryRewriter(llm=create_llm(model_name=config.REWRITER_MODEL))
t0 = time.time()
rewritten = rewriter.rewrite(query)
rewrite_total_ms = int((time.time() - t0) * 1000)

print(f"\nParsed queries ({rewrite_total_ms}ms):")
for i, q in enumerate(rewritten):
    print(f"  [{i}] text={q.text!r}  facet={q.facet_type}  filters={q.filters}  sorts={q.sorts}")


# ── 2. SEARCHER ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STAGE 2: SEARCHER")
print("─" * 70)

from search.searcher import DocumentSearcher

searcher = DocumentSearcher()

t0 = time.time()
search_results = searcher.search(rewritten, limit=5)
searcher_ms = int((time.time() - t0) * 1000)

print(f"\nSearch completed in {searcher_ms}ms")
print(f"  Results per query: {search_results.results_per_query}")
print(f"  Total found: {search_results.total_before_dedup}")
print(f"  After dedup: {search_results.total_after_dedup}")

for i, doc in enumerate(search_results.documents):
    reg_id = doc.get("regulation_id", "?")
    score = doc.get("score", 0)
    text_preview = (doc.get("text", "") or "")[:120].replace("\n", " ")
    meta = doc.get("metadata", {})
    doc_type = meta.get("type", "?")
    print(f"  [{i}] {reg_id}  score={score:.3f}  type={doc_type}")
    print(f"       {text_preview}...")


# ── 3. EVALUATOR ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STAGE 3: EVALUATOR")
print("─" * 70)

evaluator = create_evaluator()

print(f"\nModel: {config.CROSS_ENCODER_MODEL}")
print(f"Threshold: {config.EVALUATOR_THRESHOLD}")
print(f"Evaluator type: {type(evaluator).__name__}")

query_texts = [q.text for q in rewritten]
print(f"Evaluating {len(search_results.documents)} docs against {len(query_texts)} queries:")
for qt in query_texts:
    print(f"  query: {qt!r}")

t0 = time.time()
evaluated = evaluator.evaluate_multi_query(
    search_results.documents,
    query_texts,
    threshold=config.EVALUATOR_THRESHOLD,
)
eval_ms = int((time.time() - t0) * 1000)

print(f"\nEvaluation completed in {eval_ms}ms")
print(f"  Accepted: {len(evaluated)}/{len(search_results.documents)}")
for ed in evaluated:
    reg_id = ed.document.get("regulation_id", "?")
    print(f"  -> {reg_id}  score={ed.relevance_score:.1f}")


# ── 4. GENERATOR ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STAGE 4: GENERATOR")
print("─" * 70)

from search.generator.prompts import (
    GENERATOR_GROUNDED_SYSTEM_PROMPT,
    build_generator_context,
    build_generator_prompt,
)

if not evaluated:
    print("\n  No documents passed evaluation — skipping generator")
    gen_ms = 0
    ttft_ms = 0
else:
    gen_model = config.GENERATOR_MODEL or config.OLLAMA_MODEL
    gen_llm = create_llm(model_name=gen_model)

    top_evaluated = sorted(evaluated, key=lambda ed: ed.relevance_score, reverse=True)
    max_docs = config.GENERATOR_MAX_DOCS
    if max_docs > 0 and len(top_evaluated) > max_docs:
        print(f"\n  Limiting context to top {max_docs}/{len(top_evaluated)} docs by relevance")
        top_evaluated = top_evaluated[:max_docs]
    documents = [ed.document for ed in top_evaluated]
    context = build_generator_context(documents)
    user_prompt_gen = build_generator_prompt(query, context)
    system_prompt_gen = GENERATOR_GROUNDED_SYSTEM_PROMPT

    print(f"\n  Model: {gen_model}")
    print(f"  LLM type: {type(gen_llm).__name__}")
    print(f"  Max tokens: {config.LLM_MAX_TOKENS}")
    print(f"  System prompt: {len(system_prompt_gen)} chars")
    print(f"  User prompt:   {len(user_prompt_gen)} chars  ({len(user_prompt_gen.split())} words)")
    print(f"  Context (docs): {len(context)} chars  ({len(documents)} docs)")
    print(f"  Total input:   ~{len(system_prompt_gen) + len(user_prompt_gen)} chars")

    print(f"\n  System prompt:")
    print(f"    {system_prompt_gen}")
    print(f"\n  User prompt (first 500 chars):")
    print(f"    {user_prompt_gen[:500]}...")

    print(f"\n  Streaming generation:")
    t0 = time.time()
    first_token_time = None
    token_count = 0
    full_text = []

    stream = gen_llm.generate(
        prompt=user_prompt_gen,
        system_prompt=system_prompt_gen,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
            ttft_ms = int((first_token_time - t0) * 1000)
            print(f"    Time to first token: {ttft_ms}ms")
        token_count += 1
        full_text.append(chunk)

    gen_ms = int((time.time() - t0) * 1000)
    response_text = "".join(full_text)

    print(f"    Total generation: {gen_ms}ms")
    print(f"    Tokens: {token_count} chunks, {len(response_text)} chars")
    tps = token_count / (gen_ms / 1000) if gen_ms > 0 else 0
    print(f"    Speed: {tps:.1f} tokens/s")
    print(f"\n  === FULL RESPONSE ===")
    print(response_text)
    print(f"  === END RESPONSE ===")


# ── SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Mode:      {config.INFERENCE_MODE}")
print(f"  Rewriter:  {rewrite_total_ms}ms  ->  {len(rewritten)} queries  (model: {config.REWRITER_MODEL})")
print(f"  Searcher:  {searcher_ms}ms  ->  {len(search_results.documents)} docs")
print(f"  Evaluator: {eval_ms}ms  ->  {len(evaluated)} accepted  (threshold: {config.EVALUATOR_THRESHOLD})")
if evaluated:
    gen_model = config.GENERATOR_MODEL or config.OLLAMA_MODEL
    print(f"  Generator: {gen_ms}ms  (TTFT: {ttft_ms}ms, model: {gen_model})")
print(f"  Total:     {rewrite_total_ms + searcher_ms + eval_ms + gen_ms}ms")

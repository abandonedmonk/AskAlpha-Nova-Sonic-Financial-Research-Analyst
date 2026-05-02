"""
Simple RAG evaluation script for the SEC RAG pipeline using Groq.

What it does:
1. Loads questions from evals/eval_dataset.json
2. Calls tools.sec_rag.query_sec_filings for context retrieval
3. Uses Groq to generate an answer from retrieved passages
4. Uses Groq again as a judge for RAGAS-style metrics
5. Writes per-question results + aggregate metrics to a JSON file

Usage:
  python -m evals.raga_eval
  python -m evals.raga_eval --limit 5
  python -m evals.raga_eval --dataset evals/eval_dataset.json --company INTC
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from config import settings
from tools.sec_rag import query_sec_filings


DEFAULT_DATASET = Path("evals/eval_dataset.json")

# Groq throttling/retry defaults for long eval runs.
DEFAULT_GROQ_DELAY_SECONDS = 0.0
DEFAULT_GROQ_MAX_RETRIES = 6
DEFAULT_GROQ_BACKOFF_SECONDS = 6.0

_groq_call_lock = asyncio.Lock()
_last_groq_call_ts = 0.0


@dataclass
class EvalItem:
    qid: str
    category: str
    question: str
    ground_truth: str


def _load_eval_items(dataset_path: Path) -> tuple[list[EvalItem], str]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    eval_set = payload["eval_set"]
    ticker = eval_set.get("ticker", "")

    items: list[EvalItem] = []
    for q in eval_set.get("questions", []):
        items.append(
            EvalItem(
                qid=q["id"],
                category=q["category"],
                question=q["question"],
                ground_truth=q["ground_truth"],
            )
        )
    return items, ticker


def _extract_json_block(text: str) -> dict[str, Any]:
    """Best-effort extraction for model outputs that include extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


async def _groq_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    delay_seconds: float = DEFAULT_GROQ_DELAY_SECONDS,
    max_retries: int = DEFAULT_GROQ_MAX_RETRIES,
) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in your .env file.")

    url = settings.groq_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.groq_model,
        "messages": messages,
        "temperature": temperature,
    }

    timeout = httpx.Timeout(settings.note_llm_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            # Global call pacing shared across answer + judge requests.
            async with _groq_call_lock:
                global _last_groq_call_ts
                now = time.monotonic()
                wait_for = max(0.0, delay_seconds - (now - _last_groq_call_ts))
                if wait_for > 0:
                    await asyncio.sleep(wait_for)
                _last_groq_call_ts = time.monotonic()

            res = await client.post(url, headers=headers, json=body)
            if res.status_code != 429:
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()

            if attempt >= max_retries:
                res.raise_for_status()

            retry_after = res.headers.get("retry-after")
            try:
                server_wait = float(retry_after) if retry_after else 0.0
            except ValueError:
                server_wait = 0.0

            backoff_wait = DEFAULT_GROQ_BACKOFF_SECONDS * (2**attempt)
            sleep_seconds = max(server_wait, backoff_wait)
            print(
                f"    Groq rate-limited (429). Waiting {sleep_seconds:.1f}s "
                f"before retry {attempt + 1}/{max_retries}..."
            )
            await asyncio.sleep(sleep_seconds)

    raise RuntimeError("Unexpected Groq retry flow exit.")


async def _answer_question(
    question: str,
    passages: list[str],
    *,
    groq_delay_seconds: float,
    groq_max_retries: int,
) -> str:
    context = "\n\n".join(passages[:3]) if passages else "No context retrieved."

    system = (
        "You are a financial filings QA assistant. "
        "Answer only from the provided context. "
        "If context is missing or insufficient, say so briefly."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Return a concise factual answer."
    )
    return await _groq_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        delay_seconds=groq_delay_seconds,
        max_retries=groq_max_retries,
    )


async def _judge_answer(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    passages: list[str],
    groq_delay_seconds: float,
    groq_max_retries: int,
) -> dict[str, Any]:
    context = "\n\n".join(passages[:3]) if passages else "No context retrieved."

    system = (
        "You are a strict RAG evaluator. "
        "Score each metric in [0,1] where higher is better. "
        "Return JSON only with the exact keys requested."
    )
    user = (
        "Evaluate this RAG answer and return JSON with keys: "
        "answer_relevance, faithfulness, context_relevance, correctness, reason.\n\n"
        f"Question:\n{question}\n\n"
        f"Ground truth:\n{ground_truth}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Model answer:\n{answer}\n"
    )

    raw = await _groq_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        delay_seconds=groq_delay_seconds,
        max_retries=groq_max_retries,
    )
    parsed = _extract_json_block(raw)

    def _clip_score(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))

    return {
        "answer_relevance": _clip_score(parsed.get("answer_relevance")),
        "faithfulness": _clip_score(parsed.get("faithfulness")),
        "context_relevance": _clip_score(parsed.get("context_relevance")),
        "correctness": _clip_score(parsed.get("correctness")),
        "reason": str(parsed.get("reason", ""))[:400],
        "raw_judge_output": raw,
    }


async def _evaluate_one(
    item: EvalItem,
    company: str,
    filing_type: str,
    *,
    groq_delay_seconds: float,
    groq_max_retries: int,
) -> dict[str, Any]:
    rag = await query_sec_filings(
        company=company,
        topic=item.question,
        filing_type=filing_type,
    )

    passages = rag.get("passages", []) or []
    answer = await _answer_question(
        item.question,
        passages,
        groq_delay_seconds=groq_delay_seconds,
        groq_max_retries=groq_max_retries,
    )
    judge = await _judge_answer(
        question=item.question,
        answer=answer,
        ground_truth=item.ground_truth,
        passages=passages,
        groq_delay_seconds=groq_delay_seconds,
        groq_max_retries=groq_max_retries,
    )

    return {
        "id": item.qid,
        "category": item.category,
        "question": item.question,
        "ground_truth": item.ground_truth,
        "answer": answer,
        "retrieved_passages": len(passages),
        "sources": rag.get("sources", []),
        "scores": {
            "answer_relevance": judge["answer_relevance"],
            "faithfulness": judge["faithfulness"],
            "context_relevance": judge["context_relevance"],
            "correctness": judge["correctness"],
        },
        "judge_reason": judge["reason"],
        "judge_raw": judge["raw_judge_output"],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "context_relevance": 0.0,
            "correctness": 0.0,
            "overall": 0.0,
        }

    keys = ["answer_relevance", "faithfulness", "context_relevance", "correctness"]
    agg: dict[str, float] = {}
    for key in keys:
        agg[key] = mean(r["scores"][key] for r in results)
    agg["overall"] = mean(agg.values())
    return agg


async def run_eval(
    dataset_path: Path,
    company: str,
    filing_type: str,
    limit: int | None,
    output_path: Path | None,
    groq_delay_seconds: float,
    groq_max_retries: int,
) -> Path:
    items, dataset_ticker = _load_eval_items(dataset_path)
    if limit is not None:
        items = items[:limit]

    effective_company = company or dataset_ticker or "INTC"

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {item.qid}: {item.question}")
        try:
            row = await _evaluate_one(
                item,
                effective_company,
                filing_type,
                groq_delay_seconds=groq_delay_seconds,
                groq_max_retries=groq_max_retries,
            )
            results.append(row)
            c = row["scores"]["correctness"]
            f = row["scores"]["faithfulness"]
            print(f"    correctness={c:.2f} faithfulness={f:.2f}")
        except Exception as exc:
            print(f"    error: {type(exc).__name__}: {exc}")
            results.append(
                {
                    "id": item.qid,
                    "category": item.category,
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "error": str(exc),
                    "scores": {
                        "answer_relevance": 0.0,
                        "faithfulness": 0.0,
                        "context_relevance": 0.0,
                        "correctness": 0.0,
                    },
                }
            )

    summary = _aggregate(results)
    report = {
        "meta": {
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "dataset": str(dataset_path),
            "company": effective_company,
            "filing_type": filing_type,
            "total": len(items),
            "groq_model": settings.groq_model,
        },
        "summary": summary,
        "results": results,
    }

    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = Path("evals") / f"raga_results_{stamp}.json"

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple RAG evaluation with Groq")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--company", type=str, default="")
    parser.add_argument("--filing-type", type=str, default="any")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--groq-delay-seconds",
        type=float,
        default=DEFAULT_GROQ_DELAY_SECONDS,
        help="Minimum delay between Groq requests. Useful to avoid 429 on full runs.",
    )
    parser.add_argument(
        "--groq-max-retries",
        type=int,
        default=DEFAULT_GROQ_MAX_RETRIES,
        help="Retries for 429 rate limit responses.",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    out = await run_eval(
        dataset_path=args.dataset,
        company=args.company,
        filing_type=args.filing_type,
        limit=args.limit,
        output_path=args.output,
        groq_delay_seconds=max(0.0, args.groq_delay_seconds),
        groq_max_retries=max(0, args.groq_max_retries),
    )
    print("\nEvaluation complete")
    print(f"Report: {out}")


if __name__ == "__main__":
    asyncio.run(_main_async())

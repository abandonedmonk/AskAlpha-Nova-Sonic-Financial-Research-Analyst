"""
smoke_test_tools.py — Live integration tests for all 4 tool backends.

These tests make REAL network calls using your actual API keys from .env.
They are intentionally separate from the unit tests in tests/ which use mocks.

Usage:
    python scripts/smoke_test_tools.py              # test all 4 tools
    python scripts/smoke_test_tools.py market       # Tool 1 only
    python scripts/smoke_test_tools.py quant        # Tool 3 only
    python scripts/smoke_test_tools.py sec          # Tool 2 only
    python scripts/smoke_test_tools.py vault        # Tool 4 only

Exit code: 0 = all requested tests passed, 1 = any failure.
"""

import asyncio
import logging
import sys
import os
import time
import traceback
from pathlib import Path

# ── Make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Enable logging so tool internals are visible ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="  [%(name)s] %(levelname)s — %(message)s",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed: list[str] = []
failed: list[str] = []


def _ok(name: str, detail: str = "") -> None:
    passed.append(name)
    suffix = f"  {detail}" if detail else ""
    print(f"  {GREEN}✓ PASS{RESET}  {name}{suffix}")


def _fail(name: str, reason: str) -> None:
    failed.append(name)
    print(f"  {RED}✗ FAIL{RESET}  {name}")
    print(f"           {YELLOW}{reason}{RESET}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 55}{RESET}")


# ── Tool 1: market_data ───────────────────────────────────────────────────────


async def smoke_market_data() -> None:
    _section("Tool 1 — query_live_market_data  (market_data.py)")

    from tools.market_data import get_market_snapshot

    # Test A: valid ticker
    try:
        result = await get_market_snapshot("GOOGL")
        if "error" in result:
            _fail("GOOGL snapshot", f"Got error: {result['error']}")
        else:
            src = result.get("data_source", "?")
            price = result.get("price", "?")
            _ok("GOOGL snapshot", f"${price}  source={src}")
    except Exception:
        _fail("AAPL snapshot", traceback.format_exc(limit=2))

    # Test B: fallback indicator present in summary when polygon used
    try:
        result = await get_market_snapshot("NVDA")
        if "error" not in result:
            _ok(
                "NVDA snapshot",
                f"${result.get('price','?')}  source={result.get('data_source','?')}",
            )
        else:
            _fail("NVDA snapshot", result["error"])
    except Exception:
        _fail("NVDA snapshot", traceback.format_exc(limit=2))

    # Test C: invalid ticker should return an error key, not raise
    try:
        result = await get_market_snapshot("ZZZNOTREAL999")
        if "error" in result:
            _ok("invalid ticker returns error", result["error"][:80])
        else:
            _fail(
                "invalid ticker returns error",
                f"Expected error key, got: {list(result.keys())}",
            )
    except Exception:
        _fail("invalid ticker returns error", traceback.format_exc(limit=2))


# ── Tool 2: sec_rag ───────────────────────────────────────────────────────────


async def smoke_sec_rag() -> None:
    _section("Tool 2 — analyze_sec_filings_rag  (sec_rag.py)")

    from tools.sec_rag import query_sec_filings
    from config import settings

    mode = "Bedrock KB" if settings.bedrock_kb_configured else "local FAISS"
    print(f"  mode: {mode}")

    # Test A: basic query
    try:
        result = await query_sec_filings("Apple", "revenue risk", "10-K")
        passages = result.get("passages", [])
        if result.get("summary"):
            _ok("Apple revenue risk query", f"{len(passages)} passage(s) returned")
        else:
            _fail("Apple revenue risk query", "summary field missing")
    except RuntimeError as exc:
        # Expected when local FAISS index hasn't been built yet
        if (
            "faiss" in str(exc).lower()
            or "llama" in str(exc).lower()
            or "index" in str(exc).lower()
        ):
            _fail(
                "Apple revenue risk query",
                "Local FAISS index not built — run: python data/build_local_index.py",
            )
        else:
            _fail("Apple revenue risk query", str(exc))
    except Exception:
        _fail("Apple revenue risk query", traceback.format_exc(limit=2))

    # Test B: empty company name should still return a valid dict
    try:
        result = await query_sec_filings("", "test", "any")
        if isinstance(result, dict) and "summary" in result:
            _ok("empty company name graceful", "returned dict with summary")
        else:
            _fail("empty company name graceful", f"unexpected: {result}")
    except RuntimeError as exc:
        if (
            "faiss" in str(exc).lower()
            or "llama" in str(exc).lower()
            or "index" in str(exc).lower()
        ):
            _fail(
                "empty company name graceful",
                "Local FAISS index not built — run: python data/build_local_index.py",
            )
        else:
            _fail("empty company name graceful", str(exc))
    except Exception:
        _fail("empty company name graceful", traceback.format_exc(limit=2))


# ── Tool 3: quant_model ───────────────────────────────────────────────────────


async def smoke_quant_model() -> None:
    _section("Tool 3 — execute_quantitative_model  (quant_model.py)")

    from tools.quant_model import run_monte_carlo

    # Test A: standard run with production-sized path count
    try:
        started_at = time.perf_counter()
        result = await run_monte_carlo("AMD", days=30, simulations=10_000)
        elapsed_seconds = time.perf_counter() - started_at
        if "error" in result:
            _fail("AMD 30-day Monte Carlo (10,000 paths)", result["error"])
        else:
            p50 = result.get("p50", "?")
            vol_info = (
                f"price={result.get('current_price')}  p50=${p50}  "
                f"time={elapsed_seconds:.2f}s"
            )
            assert (
                result["p10"] <= result["p50"] <= result["p90"]
            ), "Percentile ordering violated!"
            _ok("AMD 30-day Monte Carlo (10,000 paths)", vol_info)
    except Exception:
        _fail("AMD 30-day Monte Carlo (10,000 paths)", traceback.format_exc(limit=2))

    # Test B: 1-day edge case
    try:
        result = await run_monte_carlo("MSFT", days=1, simulations=500)
        if "error" in result:
            _fail("MSFT 1-day edge case", result["error"])
        else:
            _ok("MSFT 1-day edge case", f"p50=${result.get('p50','?')}")
    except Exception:
        _fail("MSFT 1-day edge case", traceback.format_exc(limit=2))


# ── Tool 4: vault_logger ──────────────────────────────────────────────────────


async def smoke_vault_logger() -> None:
    _section("Tool 4 — log_research_insight  (vault_logger.py)")

    from tools.vault_logger import log_insight
    from config import settings

    print(f"  vault path: {settings.vault_path.resolve()}")
    require_note_llm = os.getenv("REQUIRE_NOTE_LLM", "0") == "1"

    # Test A: basic write
    try:
        result = await log_insight(
            content="Smoke test note from scripts/smoke_test_tools.py. Safe to delete.",
            tags=["smoke-test"],
            title="Smoke Test Note",
        )
        if result.get("saved"):
            p = Path(result["filepath"])
            exists = p.exists()
            size = p.stat().st_size if exists else 0
            provider = result.get("llm_provider", "none")
            model = result.get("llm_model", "none")
            if require_note_llm and provider == "none":
                _fail(
                    "basic write",
                    "LLM fallback path used (llm_provider=none) while REQUIRE_NOTE_LLM=1",
                )
            else:
                _ok(
                    "basic write",
                    f"file={p.name}  size={size}B  on_disk={exists}  llm={provider}:{model}",
                )
            if not exists:
                _fail(
                    "file exists on disk",
                    "log_insight returned saved=True but file not found",
                )
        else:
            _fail("basic write", f"saved=False: {result}")
    except Exception:
        _fail("basic write", traceback.format_exc(limit=2))

    # Test B: no tags, no title (should still work)
    try:
        result = await log_insight(content="Minimal smoke note. Safe to delete.")
        if result.get("saved"):
            provider = result.get("llm_provider", "none")
            model = result.get("llm_model", "none")
            if require_note_llm and provider == "none":
                _fail(
                    "write without tags/title",
                    "LLM fallback path used (llm_provider=none) while REQUIRE_NOTE_LLM=1",
                )
            else:
                _ok(
                    "write without tags/title",
                    f"{result.get('message', 'ok')}  llm={provider}:{model}",
                )
        else:
            _fail("write without tags/title", str(result))
    except Exception:
        _fail("write without tags/title", traceback.format_exc(limit=2))

    # Test C: special chars in title don't crash
    try:
        result = await log_insight(
            content="Title safety test.",
            title="Q1/2026 <Earnings> & Notes!",
        )
        if result.get("saved"):
            fname = Path(result["filepath"]).name
            provider = result.get("llm_provider", "none")
            model = result.get("llm_model", "none")
            if require_note_llm and provider == "none":
                _fail(
                    "special chars in title",
                    "LLM fallback path used (llm_provider=none) while REQUIRE_NOTE_LLM=1",
                )
            else:
                _ok(
                    "special chars in title",
                    f"filename={fname}  llm={provider}:{model}",
                )
        else:
            _fail("special chars in title", str(result))
    except Exception:
        _fail("special chars in title", traceback.format_exc(limit=2))


# ── Entry point ───────────────────────────────────────────────────────────────


async def main(targets: list[str]) -> None:
    run_all = not targets

    if run_all or "market" in targets:
        await smoke_market_data()
    if run_all or "sec" in targets:
        await smoke_sec_rag()
    if run_all or "quant" in targets:
        await smoke_quant_model()
    if run_all or "vault" in targets:
        await smoke_vault_logger()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(passed) + len(failed)
    print(f"\n{BOLD}{'═' * 55}{RESET}")
    print(
        f"{BOLD}  Results: {GREEN}{len(passed)} passed{RESET}{BOLD}  /  {RED}{len(failed)} failed{RESET}{BOLD}  /  {total} total{RESET}"
    )
    if failed:
        print(f"{BOLD}  Failed:{RESET}")
        for name in failed:
            print(f"    {RED}• {name}{RESET}")
    print(f"{BOLD}{'═' * 55}{RESET}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    targets = [a.lower() for a in sys.argv[1:]]
    asyncio.run(main(targets))

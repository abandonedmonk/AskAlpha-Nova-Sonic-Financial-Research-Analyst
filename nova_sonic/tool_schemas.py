"""
tool_schemas.py — Bedrock-compatible JSON tool schemas for Nova Sonic.

Each schema is passed to the Nova Sonic session at startup so the model
knows which tools it can call and what parameters to extract from speech.
"""

from typing import Any

# ─── Tool 1: Live market data ─────────────────────────────────────────────────
QUERY_LIVE_MARKET_DATA: dict[str, Any] = {
    "toolSpec": {
        "name": "query_live_market_data",
        "description": (
            "Fetches real-time (or latest available) price and volume data "
            "for a US equity ticker from Polygon.io. Use this whenever the "
            "user asks about current price, trading volume, or intraday "
            "price action for a stock."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": (
                            "The stock ticker symbol in uppercase, e.g. 'NVDA', "
                            "'AMD', 'AAPL'."
                        ),
                    }
                },
                "required": ["ticker"],
            }
        },
    }
}

# ─── Tool 2: SEC filings RAG ──────────────────────────────────────────────────
ANALYZE_SEC_FILINGS_RAG: dict[str, Any] = {
    "toolSpec": {
        "name": "analyze_sec_filings_rag",
        "description": (
            "Searches SEC 10-K and 10-Q filings stored in a vector knowledge "
            "base and returns the most relevant passages. Use this when the "
            "user asks what a company said about a specific topic in their "
            "annual or quarterly report."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": (
                            "Company name or ticker, e.g. 'Nvidia' or 'NVDA'."
                        ),
                    },
                    "topic": {
                        "type": "string",
                        "description": (
                            "The subject to search for in the filing, e.g. "
                            "'supply chain', 'AI revenue', 'risk factors'."
                        ),
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "any"],
                        "description": "Which filing type to search. Defaults to 'any'.",
                        "default": "any",
                    },
                },
                "required": ["company", "topic"],
            }
        },
    }
}

# ─── Tool 3: Quantitative model (Monte Carlo) ─────────────────────────────────
EXECUTE_QUANTITATIVE_MODEL: dict[str, Any] = {
    "toolSpec": {
        "name": "execute_quantitative_model",
        "description": (
            "Runs a Monte Carlo simulation using Geometric Brownian Motion to "
            "project a stock's price distribution over a future time horizon. "
            "Returns P10, P50, and P90 price percentiles plus the mean. Use "
            "this when the user asks to 'run a simulation' or 'model the price "
            "of [ticker] over the next N days'."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The equity ticker symbol.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of trading days to project forward.",
                        "minimum": 1,
                        "maximum": 252,
                    },
                    "simulations": {
                        "type": "integer",
                        "description": "Number of Monte Carlo paths (default 10 000).",
                        "default": 10000,
                    },
                },
                "required": ["ticker", "days"],
            }
        },
    }
}

# ─── Tool 4: Vault / research logger ─────────────────────────────────────────
LOG_RESEARCH_INSIGHT: dict[str, Any] = {
    "toolSpec": {
        "name": "log_research_insight",
        "description": (
            "Saves the most recent research summary as a Markdown note to the "
            "user's vault (Obsidian or local directory). Use this when the user "
            "says 'save this', 'log this', or 'add this to my vault'."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The research text or summary to persist.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of topic tags, e.g. ['semiconductors', 'nvidia'].",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional note title. Auto-generated from date if omitted.",
                    },
                },
                "required": ["content"],
            }
        },
    }
}

# ─── Aggregated list passed to the Nova Sonic session ────────────────────────
ALL_TOOLS: list[dict[str, Any]] = [
    QUERY_LIVE_MARKET_DATA,
    ANALYZE_SEC_FILINGS_RAG,
    EXECUTE_QUANTITATIVE_MODEL,
    LOG_RESEARCH_INSIGHT,
]

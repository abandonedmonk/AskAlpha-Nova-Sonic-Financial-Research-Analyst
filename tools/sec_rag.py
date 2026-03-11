"""
sec_rag.py — Tool 2: analyze_sec_filings_rag

Queries the SEC filings knowledge base.

Primary path  : AWS Bedrock Knowledge Base  (preferred — requires KB setup)
Fallback path : Local FAISS vector store    (no AWS infra needed)

The active path is selected at startup based on config.settings.bedrock_kb_configured.
If BEDROCK_KB_ID is set, the Bedrock path is used.  Otherwise the local
FAISS fallback runs (requires the sec_filings/ PDFs to have been ingested).

Bedrock KB docs:
  https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-retrieve.html
"""

import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from config import settings

logger = logging.getLogger(__name__)


# ── AWS Bedrock Knowledge Base path ───────────────────────────────────────────


async def _query_bedrock_kb(company: str, topic: str, filing_type: str) -> list[str]:
    """
    Retrieve the top-3 most relevant passages from the Bedrock Knowledge Base.
    """
    query_text = f"{company} {topic}" + (
        f" {filing_type}" if filing_type and filing_type != "any" else ""
    )

    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    try:
        logger.info(f"Bedrock KB query: '{query_text}'")
        response = client.retrieve(
            knowledgeBaseId=settings.bedrock_kb_id,
            retrievalQuery={"text": query_text},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 3}
            },
        )
    except ClientError as exc:
        logger.error("Bedrock KB retrieve failed: %s", exc)
        raise

    MIN_SCORE = 0.50  # below this the match is not meaningfully relevant

    retrieval_results = response.get("retrievalResults", [])
    logger.info(f"Bedrock returned {len(retrieval_results)} results")
    for i, result in enumerate(retrieval_results):
        source = (
            result.get("location", {})
            .get("s3Location", {})
            .get("uri", "unknown source")
        )
        score = result.get("score", "N/A")
        logger.info(f"  [{i}] source={source}  score={score}")

    passages = []
    sources_used: list[str] = []
    for result in retrieval_results:
        score = result.get("score") or 0.0
        if score < MIN_SCORE:
            logger.warning(
                "Skipping result with score %.4f (below threshold %.2f) — "
                "company '%s' may not be in the knowledge base.",
                score,
                MIN_SCORE,
                company,
            )
            continue
        text = result.get("content", {}).get("text", "")
        source = result.get("location", {}).get("s3Location", {}).get("uri", "")
        if text:
            passages.append(text.strip())
            sources_used.append(source)

    if not passages and company:
        logger.warning(
            "No passages above score threshold for company='%s'. "
            "The KB likely does not contain filings for this company.",
            company,
        )

    return passages, sources_used


# ── Local FAISS fallback path ─────────────────────────────────────────────────


async def _query_local_faiss(company: str, topic: str, filing_type: str) -> list[str]:
    """
    Fallback: query a local FAISS index built from SEC PDFs in data/sec_filings/.

    Requires:  pip install llama-index faiss-cpu pypdf
    Build the index by running:  python tools/build_local_index.py
    """
    try:
        from llama_index.core import StorageContext, load_index_from_storage  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Local FAISS fallback requires llama-index: pip install llama-index faiss-cpu pypdf"
        ) from exc

    import asyncio

    query_text = f"{company} {topic}" + (
        f" {filing_type}" if filing_type and filing_type != "any" else ""
    )

    storage_path = "./data/faiss_index"
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # llama-index query is sync; run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, query_engine.query, query_text)

    passages = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            passages.append(node.node.get_content().strip())
    else:
        passages = [str(response)]

    return passages


# ── Public tool function ──────────────────────────────────────────────────────


async def query_sec_filings(
    company: str,
    topic: str,
    filing_type: str = "any",
) -> dict[str, Any]:
    """
    Return relevant passages from SEC filings for *company* on *topic*.

    Parameters
    ----------
    company      : Company name or ticker.
    topic        : Subject to search (e.g. "supply chain", "AI revenue").
    filing_type  : "10-K", "10-Q", or "any".

    Returns
    -------
    dict compatible with the ``analyze_sec_filings_rag`` tool schema.
    """
    logger.info(
        "SEC RAG — company=%s topic=%s filing_type=%s bedrock_kb=%s",
        company,
        topic,
        filing_type,
        settings.bedrock_kb_configured,
    )

    sources_used: list[str] = []
    if settings.bedrock_kb_configured:
        passages, sources_used = await _query_bedrock_kb(company, topic, filing_type)
    else:
        logger.warning("Bedrock KB not configured — falling back to local FAISS index.")
        passages = await _query_local_faiss(company, topic, filing_type)

    if not passages:
        not_found_msg = (
            f"No relevant filings found for '{company}' on '{topic}'. "
            "This company may not be in the knowledge base."
        )
        return {
            "company": company,
            "topic": topic,
            "passages": [],
            "sources": sources_used,
            "summary": not_found_msg,
        }

    # Surface which documents were actually used so Nova Sonic can be accurate
    unique_sources = list(dict.fromkeys(
        s.split("/")[-1].replace("_", " ").replace(".pdf", "")
        for s in sources_used if s
    ))
    source_label = f" (sources: {', '.join(unique_sources)})" if unique_sources else ""

    summary = (
        f"From {company}'s filing on {topic}{source_label}: "
        + " | ".join(p[:200] for p in passages[:3])
    )

    return {
        "company": company,
        "topic": topic,
        "passages": passages,
        "sources": sources_used,
        "summary": summary,
    }

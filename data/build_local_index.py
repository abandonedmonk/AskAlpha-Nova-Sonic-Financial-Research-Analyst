"""
build_local_index.py — One-shot script to build a local FAISS vector index
from SEC 10-K/10-Q PDFs stored in data/sec_filings/.

Use this as the fallback when AWS Bedrock Knowledge Base is not configured.

Requirements:
    pip install llama-index faiss-cpu pypdf

Usage:
    python build_local_index.py
"""

import sys
from pathlib import Path

PDF_DIR = Path("data/sec_filings")
INDEX_DIR = Path("data/faiss_index")


def main() -> None:
    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext  # type: ignore
        from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore
        import faiss  # type: ignore
    except ImportError:
        print(
            "ERROR: llama-index, faiss-cpu, and pypdf are required.\n"
            "  pip install llama-index faiss-cpu pypdf",
            file=sys.stderr,
        )
        sys.exit(1)

    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(
            f"ERROR: No PDFs found in {PDF_DIR}.\n"
            "Download SEC filings from https://www.sec.gov/cgi-bin/browse-edgar and"
            " place them in data/sec_filings/.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Ingesting {len(pdfs)} PDF(s) from {PDF_DIR} ...")
    for p in pdfs:
        print(f"  {p.name}")

    documents = SimpleDirectoryReader(str(PDF_DIR)).load_data()
    print(f"Loaded {len(documents)} document chunks.")

    # 1536-dim embeddings for OpenAI ada-002 / Titan Embeddings v2
    dimension = 1536
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building index (this may take a few minutes) ...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print(f"Index saved to {INDEX_DIR}")
    print("Done. Set BEDROCK_KB_ID= (empty) in .env to use this local index.")


if __name__ == "__main__":
    main()

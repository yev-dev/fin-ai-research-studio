#!/usr/bin/env python3
"""Bulk upload documents into the RAG vector DB.

Scans a directory for files of a given type and calls
`fin_ai.core.processor.process_uploaded_document` for each file, using the
same provider/model configuration pattern as the dashboard.

Usage: python scripts/rag_bulk_upload.py --dir /path/to/files --type pdf
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fin_ai.config.fin_ai import (
    
    DEFAULT_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    OLLAMA_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
    
    SUPPORTED_SOURCE_SUFFIXES,
    LOG_DIR,
)
from fin_ai.logging_config import configure_logging
from fin_ai.core.processor import process_uploaded_document


def _matches_type(path: Path, doc_type: str) -> bool:
    suffix = path.suffix.lower()
    if doc_type.startswith("pdf"):
        return suffix == ".pdf"
    if doc_type.startswith("csv"):
        return suffix == ".csv"
    # fallback: accept any supported suffix listed in config
    return suffix in SUPPORTED_SOURCE_SUFFIXES


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk upload documents into RAG vector DB")
    parser.add_argument("--dir", required=True, help="Directory containing files to upload")
    parser.add_argument("--type", required=True, choices=["pdf", "csv", "json", "html", "docx"], help="Document type to upload")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Chat/provider name (used to select tokens/endpoints)")
    parser.add_argument("--emb-provider", default=DEFAULT_EMBEDDINGS_PROVIDER or "ollama", help="Embeddings provider")
    parser.add_argument("--emb-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model id")
    parser.add_argument("--emb-base", default=OLLAMA_BASE_URL, help="Embedding base URL")
    parser.add_argument("--github-token", default="", help="GitHub token for provider if needed")
    parser.add_argument("--deepseek-token", default="", help="DeepSeek token if needed")
    parser.add_argument("--proxy-port", type=int, default=0, help="Proxy port (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually upload, just log what would be done")
    args = parser.parse_args()

    configure_logging(log_dir=str(LOG_DIR))
    logger = logging.getLogger("rag_bulk_upload")

    input_dir = Path(args.dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", str(input_dir))
        return 2

    files = sorted(p for p in input_dir.iterdir() if p.is_file() and _matches_type(p, args.type))
    if not files:
        logger.info("No files of type %s found in %s", args.type, str(input_dir))
        return 0

    logger.info("Found %d files to upload", len(files))

    for idx, path in enumerate(files, start=1):
        fname = path.name
        logger.info("[%d/%d] Processing %s", idx, len(files), fname)
        try:
            data = path.read_bytes()
        except Exception as e:
            logger.exception("Failed to read file %s: %s", fname, e)
            continue

        if args.dry_run:
            logger.info("Dry run: would upload %s as type %s", fname, args.type)
            continue

        try:
            # Map emb provider base URL selection heuristics
            emb_base = args.emb_base or (GITHUB_EMBEDDING_BASE_URL if args.emb_provider == "github" else OLLAMA_BASE_URL)
            gh_token = args.github_token or ""

            res = process_uploaded_document(
                file_binary=data,
                file_name=fname,
                embedding_model=args.emb_model,
                embedding_base_url=emb_base,
                emb_provider=args.emb_provider,
                source_type=args.type,
                github_token=gh_token if args.emb_provider == "github" else None,
            )
            logger.info("Uploaded %s: elapsed=%.2f sec", fname, float(res.get("elapsed", 0.0)))
        except Exception as e:
            logger.exception("Error uploading %s: %s", fname, e)
            continue

    logger.info("Bulk upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

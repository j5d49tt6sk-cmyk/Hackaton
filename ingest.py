from __future__ import annotations

import argparse
import logging
from pathlib import Path

from company_brain.config import Settings
from company_brain.ingestion import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest company knowledge files into Supabase pgvector."
    )
    parser.add_argument("path", type=Path, help="File or folder to scan recursively")
    parser.add_argument("--expert", help="Override expert label for all ingested chunks")
    parser.add_argument("--topic", help="Override topic label for all ingested chunks")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Keep existing matching documents instead of replacing them",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    settings = Settings.from_env()
    pipeline = IngestionPipeline(settings)
    inserted = pipeline.ingest_path(
        args.path,
        expert=args.expert,
        topic=args.topic,
        batch_size=args.batch_size,
        replace_existing=not args.no_replace,
    )
    print(f"Inserted {inserted} chunks.")


if __name__ == "__main__":
    main()

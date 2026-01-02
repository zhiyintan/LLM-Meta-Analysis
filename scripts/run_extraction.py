#!/usr/bin/env python3
"""
Run LLM extraction on a document corpus.

Usage:
    python scripts/run_extraction.py --domain social --mode global --model gpt-5.1
    python scripts/run_extraction.py --domain agriculture --mode per-paper --model qwen3-vl
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pipeline import (
    load_questions,
    build_doc_contexts,
    run_global_extraction,
    run_map_reduce_extraction,
)
from openai import OpenAI
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM extraction on document corpus"
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["agriculture", "health", "social"],
        help="Domain to process",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["global", "per-paper"],
        help="Extraction mode: global (all docs at once) or per-paper (map-reduce)",
    )
    parser.add_argument(
        "--model", default="gpt-5.1", help="Model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Directory containing markdown documents (auto-detected from domain if not provided)",
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        default="data/queries/standardized_config.json",
        help="Path to query configuration file",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url", default=None, help="Custom API base URL (for local models)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384, help="Max tokens for LLM response"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)

    # Setup paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Auto-detect docs directory if not provided
    if args.docs_dir:
        docs_dir = Path(args.docs_dir)
    else:
        # Assume mineru_output structure
        docs_dir = Path(f"mineru_output/{args.domain}")
        if not docs_dir.exists():
            print(f"ERROR: Documents directory not found: {docs_dir}")
            print("Please provide --docs-dir or ensure mineru_output/{domain} exists")
            sys.exit(1)

    # Setup output
    output_dir = (
        Path(args.output_dir)
        / f"{args.domain}_{args.model.replace('-', '_')}_{args.mode.replace('-', '_')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    questions = load_questions(config_path)
    print(f"Loaded {len(questions)} questions from {config_path}")

    # Build document contexts
    print(f"Loading documents from {docs_dir}...")
    docs = build_doc_contexts(docs_dir)
    print(f"Loaded {len(docs)} documents")

    # Setup client
    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # Run extraction
    print(f"\nRunning {args.mode} extraction with {args.model}...")

    if args.mode == "global":
        results = run_global_extraction(
            client=client,
            model=args.model,
            docs=docs,
            questions=questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:  # per-paper
        results = run_map_reduce_extraction(
            client=client,
            model=args.model,
            docs=docs,
            questions=questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # Save results
    output_file = output_dir / "predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run evaluation on extraction predictions.

Usage:
    python scripts/run_evaluation.py --predictions results/social_gpt_global/predictions.json --ground-truth data/ground_truth/social.json
"""

import argparse
import sys
from pathlib import Path
import json
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import (
    evaluate_question,
    run_semantic_review,
)
from openai import OpenAI
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extraction predictions against ground truth"
    )
    parser.add_argument(
        "--predictions", required=True, help="Path to predictions.json file"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to predictions directory)",
    )
    parser.add_argument(
        "--semantic-review",
        action="store_true",
        help="Run semantic matching with LLM for unmatched pairs",
    )
    parser.add_argument(
        "--qwen-url",
        default="http://localhost:8000/v1",
        help="Qwen API URL for semantic review",
    )

    args = parser.parse_args()

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: Predictions file not found: {pred_path}")
        sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Load ground truth
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"ERROR: Ground truth file not found: {gt_path}")
        sys.exit(1)

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Setup output
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Question IDs to evaluate
    question_ids = [
        "T1.1",
        "T1.2",
        "T1.3",
        "T1.4",
        "T1.5",
        "T1.6",
        "T2.1",
        "T2.2",
        "T2.3",
        "T2.4",
        "T2.5",
        "T3.1",
        "T3.2",
        "T3.3",
        "T4.1",
        "T5.1",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]

    print("=" * 60)
    print("Evaluating Predictions")
    print("=" * 60)

    all_unmatched = []
    results = {"by_question": {}}

    for q_id in question_ids:
        gt_data = ground_truth.get(q_id, {})
        pred_data = predictions.get(q_id, {})

        result = evaluate_question(q_id, pred_data, gt_data)
        results["by_question"][q_id] = result

        # Collect unmatched for semantic review
        for pair in result.get("unmatched_pairs", []):
            pair["question_id"] = q_id
            all_unmatched.append(pair)

        m = result["matched"]
        p = result["prediction_count"]
        g = result["ground_truth_count"]
        prec = m / p if p > 0 else 0
        rec = m / g if g > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(
            f"  {q_id}: P={p:3d}, G={g:3d}, M={m:3d} | Prec={prec:.2f} Rec={rec:.2f} F1={f1:.2f}"
        )

    # Semantic review if requested
    llm_matches_by_q = defaultdict(int)
    if args.semantic_review and all_unmatched:
        print(f"\nRunning semantic review on {len(all_unmatched)} unmatched pairs...")
        qwen_client = OpenAI(api_key="EMPTY", base_url=args.qwen_url)

        pairs_to_review = [
            p for p in all_unmatched if 0.3 < p.get("similarity", 0) < 0.95
        ]
        print(f"  Reviewing {len(pairs_to_review)} pairs with moderate similarity")

        semantic_results = run_semantic_review(pairs_to_review, qwen_client, "all")

        for r in semantic_results:
            if r.get("llm_match"):
                llm_matches_by_q[r.get("question_id", "all")] += 1

        # Save semantic review results
        semantic_file = output_dir / "semantic_review.json"
        with open(semantic_file, "w", encoding="utf-8") as f:
            json.dump({"pairs": semantic_results}, f, indent=2, ensure_ascii=False)
        print(f"  Semantic review saved to {semantic_file}")

    # Calculate final scores
    print("\n" + "=" * 60)
    print("Final F1 Scores")
    print("=" * 60)

    final_results = {"by_question": {}, "overall": {}}
    total_matched = 0
    total_pred = 0
    total_gt = 0

    for q_id in question_ids:
        base = results["by_question"].get(q_id, {})
        m = base.get("matched", 0)
        p = base.get("prediction_count", 0)
        g = base.get("ground_truth_count", 0)

        # Add LLM matches
        m_adj = min(m + llm_matches_by_q.get(q_id, 0), p, g)

        prec = m_adj / p if p > 0 else 0
        rec = m_adj / g if g > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        final_results["by_question"][q_id] = {
            "prediction_count": p,
            "ground_truth_count": g,
            "matched": m,
            "matched_with_semantic": m_adj,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        total_matched += m_adj
        total_pred += p
        total_gt += g

        print(f"  {q_id}: F1={f1:.2f} (P={prec:.2f}, R={rec:.2f})")

    # Overall
    overall_prec = total_matched / total_pred if total_pred > 0 else 0
    overall_rec = total_matched / total_gt if total_gt > 0 else 0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0
    )

    final_results["overall"] = {
        "precision": round(overall_prec, 4),
        "recall": round(overall_rec, 4),
        "f1": round(overall_f1, 4),
    }

    print(
        f"\n  OVERALL: F1={overall_f1:.2f} (P={overall_prec:.2f}, R={overall_rec:.2f})"
    )

    # Save results
    eval_file = output_dir / "evaluation.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {eval_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Use the Qwen model to perform semantic matching detection and generate JSON files
that require manual verification.
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from difflib import SequenceMatcher
from loguru import logger

# Configuration
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

def string_similarity(a: str, b: str) -> float:
    """Compute string similarity"""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def llm_semantic_match(pred: str, gt: str) -> dict:
    """Use an LLM to determine whether two strings are semantically equivalent"""
    prompt = f"""Determine whether the following two texts refer to the same entity
or are semantically equivalent.

Text A (Prediction): {pred}
Text B (Ground Truth): {gt}

Please respond with a single JSON object in the following format:
{{"match": true/false, "reason": "brief reason"}}

Notes:
- If the two texts describe the same research object, variable, method, or concept,
  they should be considered a match even if phrased differently.
- Ignore differences in capitalization, abbreviations, and extra modifiers.
- For example, "maize plants" and "Zea mays" are a match.
- For example, "r = 0.85" and "Pearson correlation coefficient 0.85" are a match.
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in scientific literature information extraction. "
                        "Your task is to judge whether two texts are semantically equivalent. "
                        "Output JSON only."
                    ),
                },
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        # Extract JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return {
            "llm_match": result.get("match", False),
            "llm_reason": result.get("reason", "")
        }
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return {
            "llm_match": None,
            "llm_reason": f"LLM call failed: {str(e)}"
        }

def flatten_to_strings(data) -> list:
    """Flatten nested structures into a list of strings"""
    result = []
    if isinstance(data, dict):
        for v in data.values():
            result.extend(flatten_to_strings(v))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # For dictionaries, convert to a normalized string representation
                result.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
            else:
                result.append(str(item))
    else:
        result.append(str(data))
    return result

def process_question(q_id: str, pred_data, gt_data) -> list:
    """Process a single question and identify unmatched items"""
    pred_items = flatten_to_strings(pred_data)
    gt_items = flatten_to_strings(gt_data)
    
    # Identify potentially unmatched pairs
    suspicious_pairs = []
    
    # Track matched ground-truth indices
    matched_gt = set()
    
    for pred in pred_items:
        best_match = None
        best_sim = 0
        
        for i, gt in enumerate(gt_items):
            if i in matched_gt:
                continue
            sim = string_similarity(pred, gt)
            if sim > best_sim:
                best_sim = sim
                best_match = (i, gt)
        
        # Similarity between 0.2 and 0.85 may indicate semantic equivalence
        # despite string-level differences
        if best_match and 0.2 < best_sim < 0.85:
            suspicious_pairs.append({
                "prediction": pred,
                "ground_truth": best_match[1],
                "string_similarity": round(best_sim, 3),
                "question_id": q_id
            })
            matched_gt.add(best_match[0])
        elif best_sim < 0.2:
            # Completely unmatched: search for all potential matches
            for i, gt in enumerate(gt_items):
                if i not in matched_gt:
                    sim = string_similarity(pred, gt)
                    # Only keep pairs with some minimal similarity
                    if 0.1 < sim < 0.85:
                        suspicious_pairs.append({
                            "prediction": pred,
                            "ground_truth": gt,
                            "string_similarity": round(sim, 3),
                            "question_id": q_id
                        })
    
    return suspicious_pairs

def process_result_folder(folder_path: Path, gt_path: Path) -> dict:
    """Process a result folder"""
    pred_path = folder_path / "predictions.json"
    
    if not pred_path.exists():
        return {}
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    all_pairs = []
    
    for q_id in predictions.keys():
        if q_id in ground_truth:
            pairs = process_question(
                q_id,
                predictions[q_id],
                ground_truth[q_id]
            )
            all_pairs.extend(pairs)
    
    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = (p["prediction"][:100], p["ground_truth"][:100])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)
    
    logger.info(
        f"{folder_path.name}: found {len(unique_pairs)} "
        f"potentially unmatched items"
    )
    
    # Use LLM for semantic judgment
    if unique_pairs:
        logger.info(
            f"Running LLM judgment on {len(unique_pairs)} suspicious items..."
        )
        
        def check_pair(pair):
            result = llm_semantic_match(
                pair["prediction"],
                pair["ground_truth"]
            )
            pair.update(result)
            pair["human_verified"] = False  # Pending manual verification
            return pair
        
        with ThreadPoolExecutor(max_workers=500) as executor:
            futures = [
                executor.submit(check_pair, p)
                for p in unique_pairs
            ]
            results = []
            for f in as_completed(futures):
                results.append(f.result())
        
        unique_pairs = results
    
    return {
        "result_folder": folder_path.name,
        "ground_truth_file": gt_path.name,
        "total_suspicious": len(unique_pairs),
        "llm_matched": sum(
            1 for p in unique_pairs if p.get("llm_match") is True
        ),
        "llm_not_matched": sum(
            1 for p in unique_pairs if p.get("llm_match") is False
        ),
        "pairs": unique_pairs
    }

def main():
    results_dir = Path("results")
    
    # Define mappings
    configs = [
        ("agriculture_gpt_global", "ground_truth_agriculture_aggregated.json"),
        ("agriculture_gpt_per_paper", "ground_truth_agriculture_aggregated.json"),
        ("agriculture_qwen_global", "ground_truth_agriculture_aggregated.json"),
        ("agriculture_qwen_per_paper", "ground_truth_agriculture_aggregated.json"),
        ("social_gpt_global", "ground_truth_social_aggregated.json"),
        ("social_gpt_per_paper", "ground_truth_social_aggregated.json"),
        ("social_qwen_global", "ground_truth_social_aggregated.json"),
        ("social_qwen_per_paper", "ground_truth_social_aggregated.json"),
        ("health_gpt_global", "ground_truth_health_aggregated.json"),
        ("health_gpt_per_paper", "ground_truth_health_aggregated.json"),
        ("health_qwen_global", "ground_truth_health_aggregated.json"),
        ("health_qwen_per_paper", "ground_truth_health_aggregated.json"),
    ]
    
    all_results = []
    
    for folder_name, gt_file in configs:
        folder_path = results_dir / folder_name
        gt_path = Path(gt_file)
        
        if folder_path.exists() and gt_path.exists():
            logger.info(f"Processing {folder_name}...")
            result = process_result_folder(folder_path, gt_path)
            if result:
                all_results.append(result)
                
                # Save review results for this folder
                review_file = folder_path / "semantic_review.json"
                with open(review_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        result,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
                logger.info(f"Saved: {review_file}")
    
    # Aggregate statistics
    total_suspicious = sum(
        r["total_suspicious"] for r in all_results
    )
    total_llm_matched = sum(
        r["llm_matched"] for r in all_results
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Summary: found {total_suspicious} "
        f"potentially unmatched items in total"
    )
    logger.info(f"LLM judged as matched: {total_llm_matched}")
    logger.info(
        f"LLM judged as not matched: "
        f"{total_suspicious - total_llm_matched}"
    )
    logger.info(
        "Please inspect each results/*/semantic_review.json file "
        "for manual verification"
    )

if __name__ == "__main__":
    main()

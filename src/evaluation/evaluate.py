#!/usr/bin/env python3
"""
Complete Evaluation Pipeline

This script:
1. Reads a new ground truth CSV file
2. Generates ground truth JSON
3. Compares with predictions from an experiment
4. Performs semantic matching using the Qwen model
5. Outputs evaluation results with F1 scores

Usage:
    python evaluate_new_gt.py input.csv --predictions-dir results/social_qwen_per_paper
"""

import csv
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============================================================================
# PART 1: Ground Truth Generation (from convert_ground_truth.py)
# ============================================================================


def parse_list_field(value: str) -> list:
    """Parse a comma/semicolon-separated list into individual items."""
    if not value or value.strip() == "":
        return []
    items = re.split(r"[;,]", value)
    return [item.strip() for item in items if item.strip()]


def get_column(row: dict, *possible_names) -> str:
    """Get a column value with flexible name matching."""
    for name in possible_names:
        if name in row:
            return row[name]
        for k in row.keys():
            if k.lower() == name.lower():
                return row[k]
            if name.lower() in k.lower():
                return row[k]
    return ""


def parse_sample_size(value: str) -> int | None:
    """Parse sample size from various formats."""
    if not value or not value.strip():
        return None
    value = value.strip()
    value = re.sub(
        r"^(~|approx\.?|approximately|about|n\s*=\s*)", "", value, flags=re.IGNORECASE
    ).strip()
    value = re.sub(
        r"\s*(participants?|subjects?|samples?|patients?|individuals?|people|persons?).*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = value.replace(",", "")

    range_match = re.match(r"(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)", value)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return round((low + high) / 2)

    num_match = re.search(r"(\d+(?:\.\d+)?)", value)
    if num_match:
        return round(float(num_match.group(1)))
    return None


# LLM clients for extraction
_country_cache = {}
_effect_cache = {}


def extract_country_llm(geo_setting: str, client, model: str = "gpt-5.1") -> str:
    """Extract the country from a geographical setting using an LLM."""
    if not geo_setting:
        return ""
    if geo_setting in _country_cache:
        return _country_cache[geo_setting]

    prompt = f"""Extract the country name from this geographical setting.
Return ONLY the country name in English, nothing else.

Geographical setting: {geo_setting}

Country:"""

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        country = response.choices[0].message.content.strip()
        _country_cache[geo_setting] = country
        return country
    except Exception as e:
        print(f"Country extraction failed: {e}")
        return geo_setting.split(",")[-1].strip().rstrip(".")


def parse_effect_sizes_llm(effect_str: str, client, model: str = "gpt-5.1") -> list:
    """Parse an effect size string using an LLM."""
    if not effect_str or not effect_str.strip():
        return []

    cache_key = effect_str[:500]
    if cache_key in _effect_cache:
        return _effect_cache[cache_key]

    prompt = f"""Extract all correlations/effect sizes from this text.
For each, identify:
1. Variable A (usually the independent variable)
2. Variable B (usually the dependent variable)
3. Effect size (keep original format like "r = .166", "U = 14.0", etc.)

Return as a JSON array: [{{"var_a": "...", "var_b": "...", "effect": "..."}}]
If no effect sizes are found, return an empty array: []

Text: {effect_str[:2000]}

JSON:"""

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Extract effect sizes. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        if isinstance(data, list):
            effects = data
        elif isinstance(data, dict):
            effects = data.get("effects", data.get("results", []))
        else:
            effects = []

        results = []
        for eff in effects:
            if isinstance(eff, dict) and "var_a" in eff and "var_b" in eff:
                results.append(
                    {
                        "independent_variable": eff.get("var_a", ""),
                        "dependent_variable": eff.get("var_b", ""),
                        "effect_size": eff.get("effect", ""),
                    }
                )

        _effect_cache[cache_key] = results
        return results
    except Exception as e:
        print(f"Effect parsing failed: {e}")
        return []


def generate_ground_truth(input_file: str, client) -> dict:
    """Generate ground truth JSON from a CSV file."""

    # Detect delimiter
    with open(input_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","

    papers = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            papers.append(row)

    print(f"Loaded {len(papers)} papers from {input_file}")

    gt = {
        "T1.1": {"study_population": []},
        "T1.2": {"country": []},
        "T1.3": {"independent_variables": []},
        "T1.4": {"dependent_variables": []},
        "T1.5": {"sample_size": []},
        "T1.6": {"statistical_method": []},
        "T2.1": {"study_population__sample": []},
        "T2.2": {"study_population__country": []},
        "T2.3": {"independent_variable__unit": []},
        "T2.4": {"dependent_variable__unit": []},
        "T2.5": {"independent_variable__dependent_variable": []},
        "T3.1": {"study_population__sample__country": []},
        "T3.2": {"study_population__sample__independent_variable": []},
        "T3.3": {"study_population__sample__statistical_method": []},
        "T4.1": {
            "study_population__independent_variable__dependent_variable__effect_size": []
        },
        "T5.1": {
            "study_population__independent_variable__dependent_variable__statistical_method__effect_size": []
        },
    }

    study_pop_counts = defaultdict(int)
    country_counts = defaultdict(int)
    study_pop_country_counts = defaultdict(int)
    study_pop_country_gt100_counts = defaultdict(int)
    sample_sizes = []

    for paper in papers:
        study_pop = get_column(
            paper, "Study Population / Unit of Analysis", "Study Population"
        ).strip()
        geo_setting = get_column(paper, "Geographical Setting", "Geography", "Country")
        country = (
            extract_country_llm(geo_setting, client)
            if client
            else geo_setting.split(",")[-1].strip()
        )
        sample_size_str = get_column(paper, "Total Sample Size", "Sample Size", "N").strip()

        iv_str = get_column(paper, "Independent Variable", "Independent Variables", "IV")
        dv_str = get_column(paper, "Dependent Variable", "Dependent Variables", "DV")
        iv_unit = get_column(paper, "Independent Variable Unit", "IV Unit")
        dv_unit = get_column(paper, "Dependent Variable Unit", "DV Unit")
        stat_method = get_column(paper, "Correlation Test Method", "Statistical Method")
        effect_str = get_column(paper, "Effect Size (Correlation/P-value)", "Effect Size")

        ivs = parse_list_field(iv_str)
        dvs = parse_list_field(dv_str)
        stat_methods = parse_list_field(stat_method)

        # T1.1
        if study_pop and study_pop not in gt["T1.1"]["study_population"]:
            gt["T1.1"]["study_population"].append(study_pop)

        # T1.2
        if country and country not in gt["T1.2"]["country"]:
            gt["T1.2"]["country"].append(country)

        # T1.3
        for iv in ivs:
            if iv not in gt["T1.3"]["independent_variables"]:
                gt["T1.3"]["independent_variables"].append(iv)

        # T1.4
        for dv in dvs:
            if dv not in gt["T1.4"]["dependent_variables"]:
                gt["T1.4"]["dependent_variables"].append(dv)

        # T1.5
        gt["T1.5"]["sample_size"].append(sample_size_str if sample_size_str else None)

        # T1.6
        for method in stat_methods:
            if method not in gt["T1.6"]["statistical_method"]:
                gt["T1.6"]["statistical_method"].append(method)

        # T2.1
        if study_pop:
            gt["T2.1"]["study_population__sample"].append(
                {"study_population": study_pop, "sample_size": sample_size_str if sample_size_str else None}
            )

        # T2.2
        if study_pop and country:
            gt["T2.2"]["study_population__country"].append(
                {"study_population": study_pop, "country": country}
            )

        # T2.3
        if iv_unit:
            for iv in ivs:
                gt["T2.3"]["independent_variable__unit"].append(
                    {"independent_variable": iv, "independent_variable_unit": iv_unit}
                )

        # T2.4
        if dv_unit:
            for dv in dvs:
                gt["T2.4"]["dependent_variable__unit"].append(
                    {"dependent_variable": dv, "dependent_variable_unit": dv_unit}
                )

        # T2.5
        for iv in ivs:
            for dv in dvs:
                gt["T2.5"]["independent_variable__dependent_variable"].append(
                    {"independent_variable": iv, "dependent_variable": dv}
                )

        # T3.1
        if study_pop:
            gt["T3.1"]["study_population__sample__country"].append(
                {"study_population": study_pop, "sample_size": sample_size_str if sample_size_str else None, "country": country}
            )

        # T3.2
        for iv in ivs:
            gt["T3.2"]["study_population__sample__independent_variable"].append(
                {"study_population": study_pop, "sample_size": sample_size_str if sample_size_str else None, "independent_variable": iv}
            )

        # T3.3
        for method in stat_methods:
            gt["T3.3"]["study_population__sample__statistical_method"].append(
                {"study_population": study_pop, "sample_size": sample_size_str if sample_size_str else None, "statistical_method": method}
            )

        # T4.1 & T5.1
        effects = parse_effect_sizes_llm(effect_str, client) if client else []
        for eff in effects:
            gt["T4.1"]["study_population__independent_variable__dependent_variable__effect_size"].append(
                {
                    "study_population": study_pop,
                    "independent_variable": eff["independent_variable"],
                    "dependent_variable": eff["dependent_variable"],
                    "effect_size": eff["effect_size"],
                }
            )
            for method in stat_methods:
                gt["T5.1"]["study_population__independent_variable__dependent_variable__statistical_method__effect_size"].append(
                    {
                        "study_population": study_pop,
                        "independent_variable": eff["independent_variable"],
                        "dependent_variable": eff["dependent_variable"],
                        "statistical_method": method,
                        "effect_size": eff["effect_size"],
                    }
                )

        # Counting
        if study_pop:
            study_pop_counts[study_pop] += 1
        if country:
            country_counts[country] += 1
        if study_pop and country:
            study_pop_country_counts[(study_pop, country)] += 1
            parsed = parse_sample_size(sample_size_str)
            if parsed and parsed > 100:
                study_pop_country_gt100_counts[(study_pop, country)] += 1

        parsed = parse_sample_size(sample_size_str)
        if parsed:
            sample_sizes.append(parsed)

    # S queries
    gt["S1"] = {
        "study_population_counts": [
            {"study_population": sp, "paper_count": cnt} for sp, cnt in study_pop_counts.items()
        ]
    }
    gt["S2"] = {
        "country_counts": [
            {"country": c, "paper_count": cnt} for c, cnt in country_counts.items()
        ]
    }
    gt["S3"] = {
        "study_population__country_counts": [
            {"study_population": sp, "country": c, "paper_count": cnt}
            for (sp, c), cnt in study_pop_country_counts.items()
        ]
    }
    gt["S4"] = {
        "average_sample_size": round(sum(sample_sizes) / len(sample_sizes), 2) if sample_sizes else None
    }
    gt["S5"] = {
        "study_population__country_counts_gt100": [
            {"study_population": sp, "country": c, "paper_count": cnt}
            for (sp, c), cnt in study_pop_country_gt100_counts.items()
        ]
    }

    return gt


# ============================================================================
# PART 2: Evaluation & Comparison
# ============================================================================


def normalize_value(val) -> str:
    """Normalize a value for comparison."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s.-]", "", s)
    return s


def string_similarity(a: str, b: str) -> float:
    """Calculate the string similarity ratio."""
    return SequenceMatcher(None, normalize_value(a), normalize_value(b)).ratio()


def extract_numbers(s: str) -> list:
    """Extract all numbers from a string."""
    return re.findall(r"-?\d+\.?\d*", str(s))


def compare_values(pred, gt) -> tuple[bool, float]:
    """Compare prediction with ground truth. Returns (exact_match, similarity)."""
    pred_norm = normalize_value(pred)
    gt_norm = normalize_value(gt)

    if pred_norm == gt_norm:
        return True, 1.0

    sim = string_similarity(pred, gt)
    return False, sim


def evaluate_question(q_id: str, pred_data: dict, gt_data: dict) -> dict:
    """Evaluate a single question's predictions against the ground truth."""

    # Get the data arrays
    pred_key = list(pred_data.keys())[0] if pred_data else None
    gt_key = list(gt_data.keys())[0] if gt_data else None

    if not pred_key or not gt_key:
        return {
            "matched": 0,
            "prediction_count": 0,
            "ground_truth_count": 0,
            "unmatched_pairs": [],
        }

    pred_items = pred_data.get(pred_key, [])
    gt_items = gt_data.get(gt_key, [])

    # Handle simple list types (T1.x)
    if (
        isinstance(pred_items, list)
        and len(pred_items) > 0
        and not isinstance(pred_items[0], dict)
    ):
        matched = 0
        unmatched = []
        gt_matched = set()

        for p in pred_items:
            found = False
            for i, g in enumerate(gt_items):
                if i not in gt_matched:
                    exact, sim = compare_values(p, g)
                    if exact or sim > 0.85:
                        matched += 1
                        gt_matched.add(i)
                        found = True
                        break
            if not found:
                # Find the best match for unmatched pairs
                best_sim = 0
                best_gt = None
                for g in gt_items:
                    _, sim = compare_values(p, g)
                    if sim > best_sim:
                        best_sim = sim
                        best_gt = g
                unmatched.append(
                    {"prediction": str(p), "ground_truth": str(best_gt), "similarity": best_sim}
                )

        return {
            "matched": matched,
            "prediction_count": len(pred_items),
            "ground_truth_count": len(gt_items),
            "unmatched_pairs": unmatched,
        }

    # Handle dict/object types (T2.x - T5.x, S queries)
    matched = 0
    unmatched = []
    gt_matched = set()

    for p in pred_items if isinstance(pred_items, list) else [pred_items]:
        if not isinstance(p, dict):
            continue

        found = False
        for i, g in enumerate(gt_items if isinstance(gt_items, list) else [gt_items]):
            if i in gt_matched or not isinstance(g, dict):
                continue

            # Compare all fields
            all_match = True
            total_sim = 0
            field_count = 0

            for key in g.keys():
                pred_val = p.get(key)
                gt_val = g.get(key)
                exact, sim = compare_values(pred_val, gt_val)
                total_sim += sim
                field_count += 1
                if not exact and sim < 0.85:
                    all_match = False

            avg_sim = total_sim / field_count if field_count > 0 else 0

            if all_match or avg_sim > 0.9:
                matched += 1
                gt_matched.add(i)
                found = True
                break

        if not found:
            unmatched.append({"prediction": p, "ground_truth": None, "similarity": 0})

    return {
        "matched": matched,
        "prediction_count": len(pred_items) if isinstance(pred_items, list) else 1,
        "ground_truth_count": len(gt_items) if isinstance(gt_items, list) else 1,
        "unmatched_pairs": unmatched,
    }


# ============================================================================
# PART 3: Semantic Matching with Qwen
# ============================================================================


def semantic_match_qwen(pred: str, gt: str, qwen_client, question_text: str = "") -> tuple[bool, str]:
    """
    Use Qwen to determine whether the prediction semantically matches the ground truth.
    LENIENT on semantic/text matching, STRICT on numeric values.
    """

    prompt = f"""Determine whether the following two texts are a semantic match.

Important rules:
1. **Be lenient on semantic matching**:
   - Synonyms and near-synonyms should be considered a match (e.g., "sample size" ≈ "样本量" ≈ "n")
   - Abbreviations and full names should match (e.g., "UK" ≈ "United Kingdom")
   - Minor phrasing differences should be ignored (e.g., "student nurses" ≈ "nursing students")
   - Differences in casing and punctuation should be ignored

2. **Be strict on numeric comparisons**:
   - If both texts contain numbers, the numbers must match exactly
   - ".166" and "0.166" should be treated as the same
   - "-0.5" and "−0.5" (different minus signs) should be treated as the same
   - But ".166" and ".167" are different → FALSE

3. **Decision logic**:
   - If semantics match and numbers match (or neither has numbers) → TRUE
   - If semantics match but numbers differ → FALSE
   - If semantics do not match → FALSE

Prediction: {pred}
Ground truth: {gt}

Please answer:
MATCH: TRUE or FALSE
REASON: <brief reason>"""

    try:
        response = qwen_client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
            temperature=0,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()

        # Parse response
        match = False
        reason = ""

        for line in content.split("\n"):
            if "MATCH:" in line.upper():
                match = "TRUE" in line.upper()
            if "REASON:" in line.upper():
                reason = line.split(":", 1)[-1].strip()

        return match, reason

    except Exception as e:
        return False, f"Error: {e}"


def run_semantic_review(unmatched_pairs: list, qwen_client, question_id: str, max_workers: int = 10) -> list:
    """Run semantic matching on unmatched pairs using Qwen."""

    results = []

    def process_pair(pair):
        pred = str(pair.get("prediction", ""))
        gt = str(pair.get("ground_truth", ""))

        if not pred or not gt:
            return {"prediction": pred, "ground_truth": gt, "llm_match": False, "reason": "Empty value"}

        match, reason = semantic_match_qwen(pred, gt, qwen_client)
        return {
            "question_id": question_id,
            "prediction": pred,
            "ground_truth": gt,
            "string_similarity": pair.get("similarity", 0),
            "llm_match": match,
            "reason": reason,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pair, pair) for pair in unmatched_pairs]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Semantic review error: {e}")

    return results


# ============================================================================
# PART 4: Main Pipeline
# ============================================================================


def run_pipeline(csv_file: str, predictions_dir: str, output_dir: str, gpt_api_key: str, qwen_base_url: str):
    """Run the complete evaluation pipeline."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize clients
    gpt_client = OpenAI(api_key=gpt_api_key)
    qwen_client = OpenAI(api_key="EMPTY", base_url=qwen_base_url)

    print("=" * 60)
    print("STEP 1: Generating Ground Truth from CSV")
    print("=" * 60)

    gt = generate_ground_truth(csv_file, gpt_client)
    gt_file = output_path / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"Saved ground truth to: {gt_file}")

    print("\n" + "=" * 60)
    print("STEP 2: Loading Predictions")
    print("=" * 60)

    pred_file = Path(predictions_dir) / "predictions.json"
    if not pred_file.exists():
        print(f"ERROR: Predictions file not found: {pred_file}")
        return

    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"Loaded predictions from: {pred_file}")

    print("\n" + "=" * 60)
    print("STEP 3: Evaluating Questions")
    print("=" * 60)

    question_ids = [
        "T1.1", "T1.2", "T1.3", "T1.4", "T1.5", "T1.6",
        "T2.1", "T2.2", "T2.3", "T2.4", "T2.5",
        "T3.1", "T3.2", "T3.3",
        "T4.1", "T5.1",
        "S1", "S2", "S3", "S4", "S5",
    ]

    evaluation_results = {"by_question": {}}
    all_unmatched = []

    for q_id in question_ids:
        gt_data = gt.get(q_id, {})
        pred_data = predictions.get(q_id, {})

        result = evaluate_question(q_id, pred_data, gt_data)
        evaluation_results["by_question"][q_id] = result

        # Collect unmatched pairs for semantic review
        for pair in result.get("unmatched_pairs", []):
            pair["question_id"] = q_id
            all_unmatched.append(pair)

        m = result["matched"]
        p = result["prediction_count"]
        g = result["ground_truth_count"]
        prec = m / p if p > 0 else 0
        rec = m / g if g > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"  {q_id}: P={p:3d}, G={g:3d}, M={m:3d} | P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    print(f"\nTotal unmatched pairs for semantic review: {len(all_unmatched)}")

    print("\n" + "=" * 60)
    print("STEP 4: Semantic Matching with Qwen")
    print("=" * 60)

    # Filter pairs with moderate similarity for review
    pairs_to_review = [p for p in all_unmatched if 0.3 < p.get("similarity", 0) < 0.95]
    print(f"Pairs to review (0.3 < sim < 0.95): {len(pairs_to_review)}")

    semantic_results = []
    if pairs_to_review:
        semantic_results = run_semantic_review(pairs_to_review, qwen_client, "all")

        # Count matches
        llm_matches = sum(1 for r in semantic_results if r.get("llm_match"))
        print(f"LLM semantic matches: {llm_matches}/{len(semantic_results)}")

    # Save semantic review results
    semantic_file = output_path / "semantic_review.json"
    with open(semantic_file, "w", encoding="utf-8") as f:
        json.dump({"pairs": semantic_results}, f, indent=2, ensure_ascii=False)
    print(f"Saved semantic review to: {semantic_file}")

    print("\n" + "=" * 60)
    print("STEP 5: Calculating Final F1 Scores")
    print("=" * 60)

    # Build LLM match lookup
    llm_matches_by_q = defaultdict(int)
    for r in semantic_results:
        if r.get("llm_match"):
            llm_matches_by_q[r.get("question_id", "all")] += 1

    final_results = {"by_question": {}, "overall": {}}
    total_matched = 0
    total_pred = 0
    total_gt = 0

    for q_id in question_ids:
        base = evaluation_results["by_question"].get(q_id, {})
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

    print(f"\n  OVERALL: F1={overall_f1:.2f} (P={overall_prec:.2f}, R={overall_rec:.2f})")

    # Save final results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved final results to: {results_file}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Complete evaluation pipeline")
    parser.add_argument("csv_file", help="Input CSV file with ground truth data")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing predictions.json")
    parser.add_argument("--output-dir", default="./evaluation_output", help="Output directory")
    parser.add_argument(
        "--gpt-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument("--qwen-url", default="http://localhost:8000/v1", help="Qwen API base URL")

    args = parser.parse_args()

    if not args.gpt_key:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or use --gpt-key")
        exit(1)

    run_pipeline(args.csv_file, args.predictions_dir, args.output_dir, args.gpt_key, args.qwen_url)

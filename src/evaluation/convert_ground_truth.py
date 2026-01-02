#!/usr/bin/env python3
"""
Convert Ground Truth TSV/CSV to Aggregated JSON

This script reads ground truth data from a TSV/CSV file and generates
a JSON file compatible with the standardized question schema.

Columns expected:
- DOI
- Paper Title
- Study Population / Unit of Analysis
- Independent Variable
- Dependent Variable
- Geographical Setting
- Total Sample Size
- Independent Variable Unit
- Dependent Variable Unit
- Correlation Test Method
- Effect Size (Correlation/P-value)
"""

import csv
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from openai import OpenAI


def parse_list_field(value: str) -> list:
    """Parse comma/semicolon separated list into individual items."""
    if not value or value.strip() == "":
        return []
    # Split by comma or semicolon, strip whitespace
    items = re.split(r"[;,]", value)
    return [item.strip() for item in items if item.strip()]


def get_column(row: dict, *possible_names) -> str:
    """Get column value with flexible name matching."""
    for name in possible_names:
        # Exact match
        if name in row:
            return row[name]
        # Case-insensitive match
        for k in row.keys():
            if k.lower() == name.lower():
                return row[k]
            # Substring match for flexibility
            if name.lower() in k.lower():
                return row[k]
    return ""


def parse_sample_size(value: str) -> int | None:
    """Parse sample size from various formats.

    Handles:
    - Plain integers: "103" -> 103
    - With commas: "1,234" -> 1234
    - Floats: "103.5" -> 104 (rounded)
    - Approximate: "~100", "approx. 100" -> 100
    - Ranges: "100-200" -> 150 (midpoint)
    - With units: "103 participants" -> 103
    """
    if not value or not value.strip():
        return None

    value = value.strip()

    # Remove common prefixes
    value = re.sub(
        r"^(~|approx\.?|approximately|about|n\s*=\s*)", "", value, flags=re.IGNORECASE
    ).strip()

    # Remove common suffixes
    value = re.sub(
        r"\s*(participants?|subjects?|samples?|patients?|individuals?|people|persons?).*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    # Remove commas from numbers
    value = value.replace(",", "")

    # Handle ranges (take midpoint)
    range_match = re.match(r"(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)", value)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return round((low + high) / 2)

    # Extract first number
    num_match = re.search(r"(\d+(?:\.\d+)?)", value)
    if num_match:
        return round(float(num_match.group(1)))

    return None


def extract_country(geo_setting: str) -> str:
    """Extract country from geographical setting using simple heuristics."""
    if not geo_setting:
        return ""
    # Common patterns: "City, Country" or "Region, Country" or just "Country"
    # Try to get the last part after comma
    parts = geo_setting.split(",")
    country = parts[-1].strip().rstrip(".")
    # Clean up common suffixes
    country = re.sub(
        r"\s*(UK|USA|US)\.?$",
        lambda m: {
            "UK": "United Kingdom",
            "USA": "United States",
            "US": "United States",
        }.get(m.group(1), m.group(1)),
        country,
    )
    if country.upper() == "UK":
        country = "United Kingdom"
    return country


# LLM-based country extraction
_country_cache = {}


def extract_country_llm(geo_setting: str, client=None, model: str = "gpt-5.1") -> str:
    """Extract country from geographical setting using LLM."""
    if not geo_setting:
        return ""

    # Check cache
    if geo_setting in _country_cache:
        return _country_cache[geo_setting]

    if client is None:
        # Fallback to heuristic
        return extract_country(geo_setting)

    prompt = f"""Extract the country name from this geographical setting. 
Return ONLY the country name in English, nothing else.
If multiple countries, return the primary one.
If unclear, return the most likely country.

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
        print(f"LLM extraction failed for '{geo_setting}': {e}")
        return extract_country(geo_setting)


def parse_effect_sizes(effect_str: str) -> list:
    """Parse effect size string into list of (iv, dv, effect) tuples using heuristics."""
    if not effect_str:
        return []

    results = []
    # Pattern: "IV & DV: r = .XXX" or similar
    pattern = r"([^:&;]+)\s*&\s*([^:]+):\s*r\s*=\s*([-.0-9]+)"
    matches = re.findall(pattern, effect_str)
    for iv, dv, r in matches:
        results.append(
            {
                "independent_variable": iv.strip(),
                "dependent_variable": dv.strip(),
                "effect_size": f"r = {r.strip()}",
            }
        )
    return results


# LLM-based effect size parsing
_effect_cache = {}


def parse_effect_sizes_llm(
    effect_str: str, client=None, model: str = "gpt-5.1"
) -> list:
    """Parse effect size string using LLM to handle various formats."""
    if not effect_str or not effect_str.strip():
        return []

    # Check cache
    cache_key = effect_str[:500]  # Truncate for cache key
    if cache_key in _effect_cache:
        return _effect_cache[cache_key]

    if client is None:
        return parse_effect_sizes(effect_str)

    prompt = f"""Extract all correlations/effect sizes from this text.
For each, identify:
1. Variable A (usually independent variable)
2. Variable B (usually dependent variable)  
3. Effect size (keep original format like "r = .166", "U = 14.0", "t = 2.5", etc.)

Return as JSON array: [{{"var_a": "...", "var_b": "...", "effect": "..."}}]

If no effect sizes found, return empty array: []

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
                    "content": "Extract effect sizes from text. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        # Handle both array and object with array
        if isinstance(data, list):
            effects = data
        elif isinstance(data, dict):
            effects = data.get("effects", data.get("results", []))
            if not isinstance(effects, list):
                effects = [effects] if effects else []
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
        print(f"LLM effect parsing failed: {e}")
        return parse_effect_sizes(effect_str)


def convert_to_ground_truth(
    input_file: str, output_file: str, use_llm: bool = False, api_key: str = None
):
    """Convert TSV/CSV to ground truth JSON."""

    # Initialize LLM client if needed
    client = None
    if use_llm and api_key:
        client = OpenAI(api_key=api_key)
        print("Using LLM for country extraction")

    # Detect delimiter
    with open(input_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","

    # Read data
    papers = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            papers.append(row)

    print(f"Loaded {len(papers)} papers from {input_file}")

    # Initialize ground truth structure
    gt = {
        # T1.x - Single entity lists
        "T1.1": {"study_population": []},
        "T1.2": {"country": []},
        "T1.3": {"independent_variables": []},
        "T1.4": {"dependent_variables": []},
        "T1.5": {"sample_size": []},
        "T1.6": {"statistical_method": []},
        # T2.x - Binary relations
        "T2.1": {"study_population__sample": []},
        "T2.2": {"study_population__country": []},
        "T2.3": {"independent_variable__unit": []},
        "T2.4": {"dependent_variable__unit": []},
        "T2.5": {"independent_variable__dependent_variable": []},
        # T3.x - Ternary relations
        "T3.1": {"study_population__sample__country": []},
        "T3.2": {"study_population__sample__independent_variable": []},
        "T3.3": {"study_population__sample__statistical_method": []},
        # T4.x - 4-ary relations
        "T4.1": {
            "study_population__independent_variable__dependent_variable__effect_size": []
        },
        # T5.x - 5-ary relations
        "T5.1": {
            "study_population__independent_variable__dependent_variable__statistical_method__effect_size": []
        },
    }

    # Track unique values for counting
    study_pop_counts = defaultdict(int)
    country_counts = defaultdict(int)
    study_pop_country_counts = defaultdict(int)
    study_pop_country_gt100_counts = defaultdict(int)
    sample_sizes = []

    for paper in papers:
        # Extract fields using flexible matching
        study_pop = get_column(
            paper,
            "Study Population / Unit of Analysis",
            "Study Population",
            "Unit of Analysis",
        ).strip()
        geo_setting = get_column(paper, "Geographical Setting", "Geography", "Country")
        if client:
            country = extract_country_llm(geo_setting, client)
        else:
            country = extract_country(geo_setting)
        sample_size = get_column(paper, "Total Sample Size", "Sample Size", "N").strip()

        iv_str = get_column(
            paper, "Independent Variable", "Independent Variables", "IV"
        )
        dv_str = get_column(paper, "Dependent Variable", "Dependent Variables", "DV")
        iv_unit = get_column(paper, "Independent Variable Unit", "IV Unit")
        dv_unit = get_column(paper, "Dependent Variable Unit", "DV Unit")
        stat_method = get_column(
            paper, "Correlation Test Method", "Statistical Method", "Test Method"
        )
        effect_str = get_column(
            paper, "Effect Size (Correlation/P-value)", "Effect Size", "Correlation"
        )

        # Parse list fields
        ivs = parse_list_field(iv_str)
        dvs = parse_list_field(dv_str)
        stat_methods = parse_list_field(stat_method)

        # T1.1 - Study population
        if study_pop and study_pop not in gt["T1.1"]["study_population"]:
            gt["T1.1"]["study_population"].append(study_pop)

        # T1.2 - Country
        if country and country not in gt["T1.2"]["country"]:
            gt["T1.2"]["country"].append(country)

        # T1.3 - Independent variables
        for iv in ivs:
            if iv not in gt["T1.3"]["independent_variables"]:
                gt["T1.3"]["independent_variables"].append(iv)

        # T1.4 - Dependent variables
        for dv in dvs:
            if dv not in gt["T1.4"]["dependent_variables"]:
                gt["T1.4"]["dependent_variables"].append(dv)

        # T1.5 - Sample size
        gt["T1.5"]["sample_size"].append(sample_size if sample_size else None)

        # T1.6 - Statistical methods
        for method in stat_methods:
            if method not in gt["T1.6"]["statistical_method"]:
                gt["T1.6"]["statistical_method"].append(method)

        # T2.1 - Study population + sample
        if study_pop:
            gt["T2.1"]["study_population__sample"].append(
                {
                    "study_population": study_pop,
                    "sample_size": sample_size if sample_size else None,
                }
            )

        # T2.2 - Study population + country
        if study_pop and country:
            gt["T2.2"]["study_population__country"].append(
                {"study_population": study_pop, "country": country}
            )

        # T2.3 - Independent variable + unit
        if iv_unit:
            for iv in ivs:
                gt["T2.3"]["independent_variable__unit"].append(
                    {"independent_variable": iv, "independent_variable_unit": iv_unit}
                )

        # T2.4 - Dependent variable + unit
        if dv_unit:
            for dv in dvs:
                gt["T2.4"]["dependent_variable__unit"].append(
                    {"dependent_variable": dv, "dependent_variable_unit": dv_unit}
                )

        # T2.5 - Independent + Dependent variable pairs
        for iv in ivs:
            for dv in dvs:
                gt["T2.5"]["independent_variable__dependent_variable"].append(
                    {"independent_variable": iv, "dependent_variable": dv}
                )

        # T3.1 - Study pop + sample + country
        if study_pop:
            gt["T3.1"]["study_population__sample__country"].append(
                {
                    "study_population": study_pop,
                    "sample_size": sample_size if sample_size else None,
                    "country": country,
                }
            )

        # T3.2 - Study pop + sample + IV
        for iv in ivs:
            gt["T3.2"]["study_population__sample__independent_variable"].append(
                {
                    "study_population": study_pop,
                    "sample_size": sample_size if sample_size else None,
                    "independent_variable": iv,
                }
            )

        # T3.3 - Study pop + sample + stat method
        for method in stat_methods:
            gt["T3.3"]["study_population__sample__statistical_method"].append(
                {
                    "study_population": study_pop,
                    "sample_size": sample_size if sample_size else None,
                    "statistical_method": method,
                }
            )

        # T4.1 and T5.1 - Parse effect sizes
        if client:
            effects = parse_effect_sizes_llm(effect_str, client)
        else:
            effects = parse_effect_sizes(effect_str)

        for eff in effects:
            gt["T4.1"][
                "study_population__independent_variable__dependent_variable__effect_size"
            ].append(
                {
                    "study_population": study_pop,
                    "independent_variable": eff["independent_variable"],
                    "dependent_variable": eff["dependent_variable"],
                    "effect_size": eff["effect_size"],
                }
            )

            for method in stat_methods:
                gt["T5.1"][
                    "study_population__independent_variable__dependent_variable__statistical_method__effect_size"
                ].append(
                    {
                        "study_population": study_pop,
                        "independent_variable": eff["independent_variable"],
                        "dependent_variable": eff["dependent_variable"],
                        "statistical_method": method,
                        "effect_size": eff["effect_size"],
                    }
                )

        # Counting for S queries
        if study_pop:
            study_pop_counts[study_pop] += 1
        if country:
            country_counts[country] += 1
        if study_pop and country:
            study_pop_country_counts[(study_pop, country)] += 1
            parsed_size = parse_sample_size(sample_size)
            if parsed_size is not None and parsed_size > 100:
                study_pop_country_gt100_counts[(study_pop, country)] += 1

        parsed_size = parse_sample_size(sample_size)
        if parsed_size is not None:
            sample_sizes.append(parsed_size)

    # S1 - Study population counts
    gt["S1"] = {
        "study_population_counts": [
            {"study_population": sp, "paper_count": cnt}
            for sp, cnt in study_pop_counts.items()
        ]
    }

    # S2 - Country counts
    gt["S2"] = {
        "country_counts": [
            {"country": c, "paper_count": cnt} for c, cnt in country_counts.items()
        ]
    }

    # S3 - Study pop + country counts
    gt["S3"] = {
        "study_population__country_counts": [
            {"study_population": sp, "country": c, "paper_count": cnt}
            for (sp, c), cnt in study_pop_country_counts.items()
        ]
    }

    # S4 - Average sample size
    gt["S4"] = {
        "average_sample_size": round(sum(sample_sizes) / len(sample_sizes), 2)
        if sample_sizes
        else None
    }

    # S5 - Study pop + country counts (sample > 100)
    gt["S5"] = {
        "study_population__country_counts_gt100": [
            {"study_population": sp, "country": c, "paper_count": cnt}
            for (sp, c), cnt in study_pop_country_gt100_counts.items()
        ]
    }

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    print(f"Generated ground truth JSON: {output_file}")
    print(f"  T1.1 study_population: {len(gt['T1.1']['study_population'])} unique")
    print(f"  T1.2 country: {len(gt['T1.2']['country'])} unique")
    print(
        f"  T1.3 independent_variables: {len(gt['T1.3']['independent_variables'])} unique"
    )
    print(
        f"  T1.4 dependent_variables: {len(gt['T1.4']['dependent_variables'])} unique"
    )
    print(f"  T1.5 sample_size: {len(gt['T1.5']['sample_size'])} entries")
    print(
        f"  T4.1 effect_size tuples: {len(gt['T4.1']['study_population__independent_variable__dependent_variable__effect_size'])}"
    )
    print(f"  S4 average_sample_size: {gt['S4']['average_sample_size']}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Convert ground truth TSV/CSV to JSON")
    parser.add_argument("input", help="Input TSV/CSV file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM extraction (use heuristics only)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm
    if use_llm and not args.api_key:
        print("WARNING: No API key provided. Using heuristic extraction only.")
        use_llm = False

    convert_to_ground_truth(args.input, args.output, use_llm, args.api_key)

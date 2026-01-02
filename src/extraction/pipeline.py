#!/usr/bin/env python3
"""
Aggregated QA Evaluation Pipeline

Supports two aggregation modes:
1. GLOBAL: Send all papers to LLM at once, get aggregated answer directly
2. MAP-REDUCE: Extract from each paper separately, then aggregate programmatically

Output format matches ground truth:
{
  "Q01": {"study_population": ["pop1", "pop2", ...]},
  "Q02": {"country": ["country1", "country2", ...]},
  ...
}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from qa import (
    read_markdown,
    iter_markdown_files,
    _interleave_markdown_and_images,
)


# ============================================================================
# Load configuration
# ============================================================================

def load_questions(config_path: Path) -> list[dict]:
    """Load questions from standardized_config.json."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config.get("questions", [])


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground truth (aggregated format)."""
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Build document contexts
# ============================================================================

def build_doc_contexts(
    outputs_dir: Path,
    max_docs: int | None = None,
    max_chars: int | None = None,
    include_images: bool = True,
) -> list[dict]:
    """
    Build document contexts from markdown files.
    Returns list of {"file": str, "text": str, "blocks": list[dict]}.
    """
    contexts = []
    
    for md_path in sorted(iter_markdown_files(outputs_dir, skip_suffixes=[])):
        rel = str(md_path.relative_to(outputs_dir))
        text = read_markdown(md_path, max_chars)
        
        if include_images:
            blocks = _interleave_markdown_and_images(
                md_path=md_path,
                text=text,
                max_images=5,
                max_image_bytes=1_000_000,
            )
        else:
            blocks = [{"type": "text", "text": text}]
        
        contexts.append({
            "file": rel,
            "text": text,
            "blocks": blocks,
        })
        
        if max_docs and len(contexts) >= max_docs:
            break
    
    logger.info("Loaded {} documents from {}", len(contexts), outputs_dir)
    return contexts


# ============================================================================
# JSON parsing
# ============================================================================

def is_complete_json(text: str) -> bool:
    """
    Quick check if text appears to be complete JSON (balanced braces/brackets).
    Returns False for obviously truncated JSON.
    """
    text = text.strip()
    if not text:
        return False
    
    # Count braces and brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Must have balanced pairs
    if open_braces != close_braces:
        return False
    if open_brackets != close_brackets:
        return False
    
    # Must start and end properly
    if text.startswith('{') and not text.rstrip().endswith('}'):
        return False
    if text.startswith('[') and not text.rstrip().endswith(']'):
        return False
    
    return True


def parse_json_response(text: str, check_complete: bool = True) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if not text:
        return {}
    
    # Extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    
    # Find JSON object or array
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)
    elif bracket_match:
        text = bracket_match.group(0)
    
    # Early check for completeness
    if check_complete and not is_complete_json(text):
        logger.warning("Incomplete JSON detected (truncated): {}...", text[:100])
        return {}
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try common fixes
        text = text.replace("'", '"')
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON: {}", text[:200])
            return {}


# ============================================================================
# GLOBAL MODE: Ask all papers at once
# ============================================================================

def ask_global_question(
    client: OpenAI,
    model: str,
    docs: list[dict],
    question: dict,
    temperature: float = 0.1,
    max_tokens: int = 16384,
) -> dict:
    """
    Ask a question across ALL documents at once.
    Returns the aggregated answer directly from LLM.
    Supports multimodal input (text + images).
    """
    q_id = question["id"]
    q_text = question["question"]
    
    system_prompt = (
        "You are a scientific literature analyst extracting aggregated data across multiple papers.\n"
        "Read ALL the papers below (including figures and tables) and extract the requested information.\n"
        "Combine and deduplicate information across all papers.\n"
        "Output your answer as a single JSON object with the aggregated list.\n"
        "Be concise but complete. Do not miss any paper."
    )
    
    # Build multimodal content blocks
    user_content = []
    user_content.append({"type": "text", "text": f"The following are {len(docs)} scientific papers:\n"})
    
    total_chars = 0
    total_images = 0
    
    for i, doc in enumerate(docs, 1):
        # Add paper header
        user_content.append({"type": "text", "text": f"\n=== PAPER {i}: {doc['file']} ===\n"})
        
        # Add blocks (text and images)
        for block in doc.get("blocks", []):
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["text"]})
                total_chars += len(block["text"])
            elif block["type"] == "image_url":
                user_content.append(block)
                total_images += 1
    
    # Add question
    user_content.append({
        "type": "text", 
        "text": f"\n\nQuestion ({q_id}): {q_text}\n\nProvide your answer as JSON only, no explanation."
    })
    
    estimated_tokens = total_chars // 4
    logger.info("{}: Input ~{} chars (~{}K tokens estimate), {} images", q_id, total_chars, estimated_tokens // 1000, total_images)
    
    token_param = {}
    if model.startswith("gpt"):
        token_param = {
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    else:
        token_param = {"max_tokens": max_tokens}
    
    # Retry logic for JSON parsing failures
    max_retries = 5
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                **token_param,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            
            message = response.choices[0].message.content or ""
            
            # Log token usage from API
            if hasattr(response, 'usage') and response.usage:
                logger.info("{}: prompt_tokens={}, completion_tokens={}, total={}", 
                           q_id, response.usage.prompt_tokens, 
                           response.usage.completion_tokens, response.usage.total_tokens)
            
            result = parse_json_response(message)
            
            # Check if result is valid
            if result:
                return result
            else:
                last_error = "Empty JSON result"
                logger.warning("{} attempt {}: {}", q_id, attempt + 1, last_error)
        
        except Exception as e:
            last_error = str(e)
            logger.warning("{} attempt {} failed: {}", q_id, attempt + 1, last_error)
    
    logger.error("Failed {} after {} retries: {}", q_id, max_retries, last_error)
    return {}


def run_global_extraction(
    client: OpenAI,
    model: str,
    docs: list[dict],
    questions: list[dict],
    temperature: float,
    max_tokens: int,
    parallel: int = 3,
) -> dict:
    """Run global extraction for all questions."""
    results = {}
    
    logger.info("Running GLOBAL extraction for {} questions across {} papers", 
                len(questions), len(docs))
    
    def ask_q(q):
        return q["id"], ask_global_question(client, model, docs, q, temperature, max_tokens)
    
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_q = {executor.submit(ask_q, q): q["id"] for q in questions}
            for future in concurrent.futures.as_completed(future_to_q):
                q_id = future_to_q[future]
                try:
                    result_id, result_data = future.result()
                    results[result_id] = result_data
                    logger.info("Completed {}: {} items", result_id, 
                               sum(len(v) if isinstance(v, list) else 1 for v in result_data.values()))
                except Exception as e:
                    logger.exception("Failed {}: {}", q_id, e)
                    results[q_id] = {}
    else:
        for q in questions:
            q_id, result = ask_q(q)
            results[q_id] = result
    
    return results


# ============================================================================
# MAP-REDUCE MODE: Extract per paper, then aggregate
# ============================================================================

def ask_paper_question(
    client: OpenAI,
    model: str,
    doc: dict,
    question: dict,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> dict:
    """Ask a question for a single paper. Supports multimodal (text + images)."""
    q_id = question["id"]
    q_text = question["question"]
    
    system_prompt = (
        "You are a scientific paper analyst extracting structured data from a single paper.\n"
        "Answer the question based ONLY on the provided paper.\n"
        "Output your answer as valid JSON matching the specified format."
    )
    
    # Build user content - use blocks for multimodal if available
    blocks = doc.get("blocks", [])
    
    if blocks and any(b.get("type") == "image_url" for b in blocks):
        # Multimodal: text + images
        user_content = [
            {"type": "text", "text": f"Paper: {doc['file']}\n\n"},
        ]
        
        for block in blocks:
            if block.get("type") == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url":
                user_content.append({
                    "type": "image_url",
                    "image_url": block["image_url"]
                })
        
        user_content.append({
            "type": "text", 
            "text": f"\n\nQuestion ({q_id}): {q_text}\n\nProvide your answer as JSON only."
        })
    else:
        # Text only
        user_content = (
            f"Paper: {doc['file']}\n\n"
            f"{doc['text']}\n\n"
            f"Question ({q_id}): {q_text}\n\n"
            "Provide your answer as JSON only."
        )
    
    token_param = {}
    if model.startswith("gpt"):
        token_param = {
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    else:
        token_param = {"max_tokens": max_tokens}
    
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            **token_param,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        
        message = response.choices[0].message.content or ""
        
        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            logger.debug("{} [{}]: prompt={}, completion={}, total={}", 
                        q_id, doc['file'][:20], response.usage.prompt_tokens, 
                        response.usage.completion_tokens, response.usage.total_tokens)
        
        return parse_json_response(message)
    
    except Exception as e:
        logger.error("Error on {} for {}: {}", q_id, doc['file'], e)
        return {}


def aggregate_results(
    client: OpenAI,
    model: str,
    per_paper_results: dict[str, dict],
    questions: list[dict],
    temperature: float = 0.1,
    parallel: int = 100,
) -> dict:
    """
    Use LLM to aggregate per-paper results into deduplicated lists.
    Processes questions in parallel.
    """
    logger.info("Aggregating results from {} papers using LLM (parallel={})...", len(per_paper_results), parallel)
    
    def aggregate_single_question(q):
        q_id = q["id"]
        q_text = q.get("original_question", q["question"])
        
        # Collect all answers for this question
        paper_answers = []
        for paper_file, answers in per_paper_results.items():
            if q_id in answers and answers[q_id]:
                paper_answers.append({
                    "paper": paper_file,
                    "answer": answers[q_id]
                })
        
        if not paper_answers:
            return q_id, {}
        
        system_prompt = """You are a data aggregation assistant.

Your task: Combine answers from multiple papers into a single aggregated result.

IMPORTANT: The input answers may have inconsistent or incorrect JSON schemas. 
Some answers may use different key names or structures. You must:
1. Understand the semantic meaning despite schema differences
2. Normalize all values into the correct output format as specified in the question
3. Merge all values into lists, removing duplicates (case-insensitive)
4. For complex nested structures, merge matching items based on content, not just structure

Think step by step, then output your final aggregated JSON after "===AGGREGATED===".

Format:
[Your thinking...]
===AGGREGATED===
{...valid JSON only, no trailing text...}"""
        
        answers_text = json.dumps(paper_answers, indent=2, ensure_ascii=False)
        
        user_content = f"""Question: {q_text}

Answers from {len(paper_answers)} papers (NOTE: schemas may vary):
{answers_text}

Aggregate these into a single JSON result matching the question's expected output format.
Make sure your JSON is complete and properly closed."""
        
        # Retry logic for JSON parsing failures
        max_retries = 5
        last_error = None
        
        for attempt in range(max_retries):
            try:
                token_param = {"max_tokens": 4096} if not model.startswith("gpt") else {"max_completion_tokens": 4096}
                
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    **token_param,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                
                full_response = response.choices[0].message.content.strip()
                
                # Log token usage for aggregation
                if hasattr(response, 'usage') and response.usage:
                    logger.debug("Aggregate {}: prompt={}, completion={}", 
                               q_id, response.usage.prompt_tokens, response.usage.completion_tokens)
                
                delimiter = "===AGGREGATED==="
                if delimiter in full_response:
                    json_part = full_response.split(delimiter)[-1].strip()
                else:
                    json_part = full_response
                
                result = parse_json_response(json_part)
                
                # Check if result is valid (not empty when we have answers)
                if result or not paper_answers:
                    logger.debug("Aggregated {}: {} items (attempt {})", q_id, len(result), attempt + 1)
                    return q_id, result
                else:
                    last_error = "Empty result despite having answers"
                    logger.warning("{} attempt {}: {}", q_id, attempt + 1, last_error)
                    
            except Exception as e:
                last_error = str(e)
                logger.warning("{} attempt {} failed: {}", q_id, attempt + 1, last_error)
        
        logger.error("Failed to aggregate {} after {} retries: {}", q_id, max_retries, last_error)
        return q_id, {}
    
    aggregated = {}
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(aggregate_single_question, q): q["id"] for q in questions}
            for future in concurrent.futures.as_completed(futures):
                q_id, result = future.result()
                aggregated[q_id] = result
    else:
        for q in questions:
            q_id, result = aggregate_single_question(q)
            aggregated[q_id] = result
    
    return aggregated


def decompose_single_question(
    client: OpenAI,
    model: str,
    question: dict,
    temperature: float = 0.1,
) -> str:
    """
    Use LLM to transform one aggregated question into a single-paper question.
    Uses chain-of-thought: LLM thinks first, then outputs result after delimiter.
    """
    q_id = question["id"]
    q_text = question["question"]
    
    system_prompt = """You are a question rewriting assistant.

Your task: Rewrite the given question from "multi-paper aggregation" format to "single-paper extraction" format.

Rules:
1. Change "List all..." to "Extract the..." or "Identify the..."
2. Change the JSON output format from a list to a single value:
   - {"key": ["<val>", ...]} becomes {"key": "<value>"}
   - {"key": [{"a": "<a>", "b": "<b>"}, ...]} becomes {"key": {"a": "<a>", "b": "<b>"}}
3. Keep the same JSON key names exactly

Examples:

BEFORE: List all study populations. Output as JSON: {"study_population": ["<pop1>", ...]}
AFTER: Extract the study population from this paper. Output as JSON: {"study_population": "<study_population>"}

BEFORE: List all study populations and sample sizes. Output as JSON: {"data": [{"pop": "<p>", "size": "<s>"}, ...]}
AFTER: Extract the study population and sample size. Output as JSON: {"data": {"pop": "<population>", "size": "<sample_size>"}}

First, think step by step about how to rewrite the question.
Then, output your final rewritten question after the delimiter "===REWRITTEN===".

Format:
[Your thinking here...]
===REWRITTEN===
[The rewritten question only, no quotes]"""
    
    user_content = f"Rewrite this question for single-paper extraction:\n\n{q_text}"
    
    try:
        token_param = {"max_tokens": 1024} if not model.startswith("gpt") else {"max_completion_tokens": 1024}
        
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            **token_param,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Extract the rewritten question after the delimiter
        delimiter = "===REWRITTEN==="
        if delimiter in full_response:
            rewritten = full_response.split(delimiter)[-1].strip()
        else:
            # Fallback: use the last line or full response
            rewritten = full_response.strip()
        
        logger.debug("Decomposed {}: {} -> {}", q_id, q_text[:50], rewritten[:50])
        return rewritten
        
    except Exception as e:
        logger.warning("Failed to decompose {}: {}, using original", q_id, e)
        return q_text


def decompose_questions_for_single_paper(
    client: OpenAI,
    model: str,
    questions: list[dict],
    temperature: float = 0.1,
    parallel: int = 100,
) -> list[dict]:
    """
    Transform all aggregated questions into single-paper questions.
    Processes questions in parallel for speed.
    """
    logger.info("Decomposing {} questions into per-paper format (parallel={})...", len(questions), parallel)
    
    def decompose_q(q):
        rewritten_text = decompose_single_question(client, model, q, temperature)
        new_q = q.copy()
        new_q["question"] = rewritten_text
        new_q["original_question"] = q["question"]
        return q["id"], new_q
    
    results = {}
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(decompose_q, q): q["id"] for q in questions}
            for future in concurrent.futures.as_completed(futures):
                try:
                    q_id, new_q = future.result()
                    results[q_id] = new_q
                except Exception as e:
                    q_id = futures[future]
                    logger.warning("Failed to decompose {}: {}", q_id, e)
                    # Keep original
                    for q in questions:
                        if q["id"] == q_id:
                            results[q_id] = q
                            break
    else:
        for q in questions:
            _, new_q = decompose_q(q)
            results[q["id"]] = new_q
    
    # Maintain original order
    new_questions = [results[q["id"]] for q in questions if q["id"] in results]
    
    logger.info("Successfully decomposed {} questions", len(new_questions))
    return new_questions


def run_map_reduce_extraction(
    client: OpenAI,
    model: str,
    docs: list[dict],
    questions: list[dict],
    temperature: float,
    max_tokens: int,
    parallel_docs: int = 3,
    parallel_questions: int = 5,
) -> dict:
    """Run map-reduce extraction: per-paper then aggregate."""
    
    logger.info("Running MAP-REDUCE extraction: {} questions × {} papers", 
                len(questions), len(docs))
    
    # Step 1: Decompose aggregated questions into per-paper format
    per_paper_questions = decompose_questions_for_single_paper(client, model, questions, temperature)
    
    per_paper_results = {}
    
    # Step 2: Process each paper with the decomposed questions
    def process_paper(doc):
        paper_results = {}
        for q in per_paper_questions:
            result = ask_paper_question(client, model, doc, q, temperature, max_tokens)
            paper_results[q["id"]] = result
        return doc["file"], paper_results
    
    if parallel_docs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_docs) as executor:
            future_to_doc = {executor.submit(process_paper, doc): doc["file"] for doc in docs}
            for future in concurrent.futures.as_completed(future_to_doc):
                try:
                    paper_file, results = future.result()
                    per_paper_results[paper_file] = results
                    logger.info("Completed paper: {}", paper_file)
                except Exception as e:
                    logger.exception("Failed paper: {}", e)
    else:
        for doc in docs:
            paper_file, results = process_paper(doc)
            per_paper_results[paper_file] = results
    
    # Aggregate using LLM
    logger.info("Aggregating results from {} papers...", len(per_paper_results))
    aggregated = aggregate_results(client, model, per_paper_results, questions, temperature)
    
    return aggregated


# ============================================================================
# Evaluation
# ============================================================================

def fuzzy_match(pred: str, gt: str, threshold: float = 0.8) -> bool:
    """Check if two strings match (exact or fuzzy)."""
    if not pred or not gt:
        return False
    
    p = str(pred).strip().lower()
    g = str(gt).strip().lower()
    
    if p == g:
        return True
    
    # Country normalization
    country_map = {
        "united states": "usa", "united states of america": "usa", "us": "usa",
        "united kingdom": "uk", "great britain": "uk", "england": "uk",
        "the netherlands": "netherlands",
    }
    p_norm = country_map.get(p, p)
    g_norm = country_map.get(g, g)
    if p_norm == g_norm:
        return True
    
    # Fuzzy match
    ratio = SequenceMatcher(None, p, g).ratio()
    return ratio >= threshold


def evaluate_list(pred_list: list, gt_list: list) -> dict:
    """Evaluate a predicted list against ground truth list."""
    if not pred_list and not gt_list:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0, "missed": 0, "extra": 0}
    
    if not pred_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "missed": len(gt_list), "extra": 0}
    
    if not gt_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "missed": 0, "extra": len(pred_list)}
    
    # Find matches
    matched_pred = set()
    matched_gt = set()
    
    for i, p in enumerate(pred_list):
        for j, g in enumerate(gt_list):
            if j not in matched_gt:
                if fuzzy_match(str(p), str(g)):
                    matched_pred.add(i)
                    matched_gt.add(j)
                    break
    
    matched = len(matched_gt)
    precision = matched / len(pred_list) if pred_list else 0
    recall = matched / len(gt_list) if gt_list else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "missed": len(gt_list) - matched,
        "extra": len(pred_list) - matched,
    }


def dict_fuzzy_match(pred_dict: dict, gt_dict: dict, threshold: float = 0.7) -> float:
    """
    Compare two dicts and return a match score (0-1).
    Uses fuzzy matching for string values.
    """
    if not pred_dict or not gt_dict:
        return 0.0
    
    # Get common keys (excluding paper_id)
    pred_keys = set(k for k in pred_dict.keys() if k != "paper_id")
    gt_keys = set(k for k in gt_dict.keys() if k != "paper_id")
    
    if not gt_keys:
        return 1.0 if not pred_keys else 0.0
    
    matched_score = 0.0
    total_keys = len(gt_keys)
    
    for k in gt_keys:
        gt_val = gt_dict.get(k)
        pred_val = pred_dict.get(k)
        
        if pred_val is None:
            # Try to find by similar key name
            for pk in pred_keys:
                if pk.lower().replace("_", "") == k.lower().replace("_", ""):
                    pred_val = pred_dict.get(pk)
                    break
        
        if pred_val is None:
            continue
        
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            # Numeric comparison with tolerance
            if gt_val == 0:
                matched_score += 1.0 if pred_val == 0 else 0.0
            else:
                diff = abs(gt_val - pred_val) / abs(gt_val)
                matched_score += max(0, 1 - diff)
        elif isinstance(gt_val, str) and isinstance(pred_val, str):
            # String fuzzy comparison
            if fuzzy_match(pred_val, gt_val, threshold):
                matched_score += 1.0
            else:
                ratio = SequenceMatcher(None, str(pred_val).lower().strip(), 
                                       str(gt_val).lower().strip()).ratio()
                if ratio >= 0.5:
                    matched_score += ratio
        else:
            # Compare as strings
            if str(gt_val).lower().strip() == str(pred_val).lower().strip():
                matched_score += 1.0
    
    return matched_score / total_keys if total_keys > 0 else 0.0


def evaluate_list_of_dicts(pred_list: list, gt_list: list, match_threshold: float = 0.6) -> dict:
    """
    Evaluate a list of predicted dicts against ground truth dicts.
    Uses dict_fuzzy_match to find best matches.
    """
    if not pred_list and not gt_list:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0, "missed": 0, "extra": 0}
    
    if not pred_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "missed": len(gt_list), "extra": 0}
    
    if not gt_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "missed": 0, "extra": len(pred_list)}
    
    # Find best matches
    matched_pred = set()
    matched_gt = set()
    
    for i, pred in enumerate(pred_list):
        best_match_idx = -1
        best_match_score = 0.0
        
        for j, gt in enumerate(gt_list):
            if j in matched_gt:
                continue
            
            score = dict_fuzzy_match(pred, gt) if isinstance(pred, dict) and isinstance(gt, dict) else 0.0
            
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j
        
        if best_match_score >= match_threshold and best_match_idx >= 0:
            matched_pred.add(i)
            matched_gt.add(best_match_idx)
    
    matched = len(matched_gt)
    precision = matched / len(pred_list) if pred_list else 0
    recall = matched / len(gt_list) if gt_list else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "missed": len(gt_list) - matched,
        "extra": len(pred_list) - matched,
    }


def extract_values_from_prediction(pred_data: Any) -> list:
    """
    Extract all values from prediction data, handling various formats:
    
    For simple questions (Q01-Q06):
    - Ground truth: {"study_population": ["pop1", "pop2"]} -> ["pop1", "pop2"]
    - Predictions: [{"paper_id": "x", "study_population": "pop1"}, ...] -> ["pop1", ...]
    
    For complex questions (Q07-Q16):
    - Ground truth: {"key": [{"field1": "a", "field2": "b"}, ...]} -> [{"field1": "a", "field2": "b"}, ...]
    - Predictions: [{"paper_id": "x", "mapping": {"field1": "a", "field2": "b"}}, ...] -> [{"field1": "a", "field2": "b"}, ...]
    """
    if pred_data is None:
        return []
    
    # If it's a list (per-paper format from predictions)
    if isinstance(pred_data, list):
        values = []
        for item in pred_data:
            if isinstance(item, dict):
                # Skip paper_id and extract actual values
                for k, v in item.items():
                    if k == "paper_id":
                        continue
                    if isinstance(v, list):
                        # List of values - extend
                        values.extend(v)
                    elif isinstance(v, dict):
                        # Nested dict (complex question like Q07-Q16) - add the whole dict
                        values.append(v)
                    elif v is not None:
                        values.append(v)
            elif item is not None:
                values.append(item)
        return values
    
    # If it's a dict (ground truth format)
    if isinstance(pred_data, dict):
        values = []
        for v in pred_data.values():
            if isinstance(v, list):
                # Check if it's a list of dicts (complex format)
                if v and isinstance(v[0], dict):
                    # Keep the list of dicts as-is for complex comparisons
                    return v
                else:
                    # Simple list of values
                    values.extend(v)
            elif v is not None:
                values.append(v)
        return values
    
    return [pred_data] if pred_data is not None else []


def evaluate_predictions(predictions: dict, ground_truth: dict) -> dict:
    """Evaluate all predictions against ground truth."""
    results = {"by_question": {}}
    
    all_q_ids = sorted(set(predictions.keys()) | set(k for k in ground_truth.keys() if k.startswith("Q")))
    
    for q_id in all_q_ids:
        pred_data = predictions.get(q_id, {})
        gt_data = ground_truth.get(q_id, {})
        
        # Extract values using the new flexible extractor
        pred_list = extract_values_from_prediction(pred_data)
        gt_list = extract_values_from_prediction(gt_data)
        
        # Check if we're dealing with complex types (list of dicts)
        is_complex_gt = gt_list and isinstance(gt_list[0], dict)
        is_complex_pred = pred_list and isinstance(pred_list[0], dict)
        
        if is_complex_gt or is_complex_pred:
            # For complex types (Q07-Q16), compare dicts
            eval_result = evaluate_list_of_dicts(pred_list, gt_list)
        else:
            # For simple lists, use string comparison
            eval_result = evaluate_list(pred_list, gt_list)
        
        results["by_question"][q_id] = {
            "prediction_count": len(pred_list),
            "ground_truth_count": len(gt_list),
            **eval_result,
        }
    
    # Summary
    f1_scores = [r["f1"] for r in results["by_question"].values()]
    results["summary"] = {
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        "avg_precision": sum(r["precision"] for r in results["by_question"].values()) / len(f1_scores) if f1_scores else 0,
        "avg_recall": sum(r["recall"] for r in results["by_question"].values()) / len(f1_scores) if f1_scores else 0,
        "total_questions": len(results["by_question"]),
    }
    
    return results


def print_evaluation_report(results: dict):
    """Print evaluation report to console."""
    print("\n" + "=" * 80)
    print("AGGREGATED EVALUATION REPORT")
    print("=" * 80)
    print(f"Average F1:        {results['summary']['avg_f1']:.3f}")
    print(f"Average Precision: {results['summary']['avg_precision']:.3f}")
    print(f"Average Recall:    {results['summary']['avg_recall']:.3f}")
    print(f"Questions:         {results['summary']['total_questions']}")
    print("\n" + "-" * 80)
    print(f"{'Q_ID':<6} {'Pred':>6} {'GT':>6} {'Match':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 80)
    
    for q_id, r in sorted(results["by_question"].items()):
        icon = "✓" if r["f1"] >= 0.8 else ("◐" if r["f1"] >= 0.5 else "✗")
        print(f"{icon} {q_id:<4} {r['prediction_count']:>6} {r['ground_truth_count']:>6} {r['matched']:>6} "
              f"{r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}")


def save_csv_report(results: dict, predictions: dict, ground_truth: dict, output_path: Path):
    """Save detailed CSV report."""
    rows = []
    
    for q_id in sorted(results["by_question"].keys()):
        r = results["by_question"][q_id]
        pred_data = predictions.get(q_id, {})
        gt_data = ground_truth.get(q_id, {})
        
        # Get first key's values
        pred_list = list(pred_data.values())[0] if pred_data else []
        gt_list = list(gt_data.values())[0] if gt_data else []
        
        if not isinstance(pred_list, list):
            pred_list = [pred_list] if pred_list else []
        if not isinstance(gt_list, list):
            gt_list = [gt_list] if gt_list else []
        
        row = {
            "question_id": q_id,
            "prediction_count": len(pred_list),
            "ground_truth_count": len(gt_list),
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "predictions": json.dumps(pred_list, ensure_ascii=False),
            "ground_truth": json.dumps(gt_list, ensure_ascii=False),
        }
        rows.append(row)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info("Saved CSV report to {}", output_path)


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(
    outputs_dir: Path,
    questions_config: Path,
    ground_truth_path: Path,
    client: OpenAI,
    model: str,
    mode: str = "global",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_chars: int = 40000,
    max_docs: int | None = None,
    include_images: bool = True,
    parallel_questions: int = 3,
    parallel_docs: int = 2,
    output_dir: Path | None = None,
) -> dict:
    """
    Run the full pipeline.
    
    Args:
        mode: "global" (all papers at once) or "map-reduce" (per-paper then aggregate)
    """
    
    # Load questions
    questions = load_questions(questions_config)
    logger.info("Loaded {} questions from {}", len(questions), questions_config)
    
    # Load documents
    docs = build_doc_contexts(outputs_dir, max_docs, max_chars, include_images)
    
    # Run extraction
    if mode == "global":
        predictions = run_global_extraction(
            client, model, docs, questions, temperature, max_tokens, parallel_questions
        )
    else:  # map-reduce
        predictions = run_map_reduce_extraction(
            client, model, docs, questions, temperature, max_tokens, parallel_docs, parallel_questions
        )
    
    # Load ground truth and evaluate
    ground_truth = load_ground_truth(ground_truth_path)
    results = evaluate_predictions(predictions, ground_truth)
    
    # Output directory
    save_dir = output_dir or outputs_dir.parent / f"results_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_path = save_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Saved predictions to {}", pred_path)
    
    # Save evaluation
    eval_path = save_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation to {}", eval_path)
    
    # Save CSV
    csv_path = save_dir / "evaluation.csv"
    save_csv_report(results, predictions, ground_truth, csv_path)
    
    # Print report
    print_evaluation_report(results)
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregated QA Evaluation Pipeline"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        required=True,
        help="Directory containing MinerU markdown exports",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent / "standardized_config.json",
        help="Path to standardized questions config",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to aggregated ground truth JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "map-reduce"],
        default="global",
        help="Extraction mode: global (all papers at once) or map-reduce (per-paper then aggregate)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-chars", type=int, default=40000)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--no-images", action="store_true", help="Disable image inclusion (images included by default)")
    parser.add_argument("--parallel-questions", type=int, default=3)
    parser.add_argument("--parallel-docs", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key, base_url=args.api_base)
    
    results = run_pipeline(
        outputs_dir=args.outputs_dir,
        questions_config=args.questions,
        ground_truth_path=args.ground_truth,
        client=client,
        model=args.model,
        mode=args.mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
        max_docs=args.max_docs,
        include_images=not args.no_images,
        parallel_questions=args.parallel_questions,
        parallel_docs=args.parallel_docs,
        output_dir=args.output_dir,
    )
    
    print(f"\n{'=' * 70}")
    print(f"FINAL SCORE (F1): {results['summary']['avg_f1']:.3f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

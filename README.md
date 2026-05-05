# LLM-Meta-Analysis

> 👉 **For the codebase and datasets, please go to the active repository:**
> **https://github.com/zhiyintan/llm-scientific-evidence-extraction**

**Paper:** *Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis* — [arxiv.org/abs/2602.10881](https://arxiv.org/abs/2602.10881)
**Venue:** IRCDL 2026, Modena, Italy, February 19–20, 2026
**Slides:** [presentation/ircdl 2026.pdf](presentation/ircdl%202026.pdf)

## TL;DR

State-of-the-art LLMs can recognize the right entities in scientific papers, but **they cannot reliably bind those entities into the role-aware tuples that meta-analysis requires**. We test this with a corpus of **41 empirical papers across agriculture, health, and social science**, evaluated under both per-document and long-context multi-document input. Findings:

- Single-property extraction is moderate.
- Performance collapses once a query requires stable binding across variables, methods, and effect sizes.
- Full association tuples are extracted with **near-zero reliability**.
- Long-context multi-document input makes things **worse**, not better.
- Failures are **structural** — role reversals, cross-analysis binding drift, instance compression in dense result sections, numeric misattribution — not entity recognition errors.

## Quick start

The query suite is defined in [data/queries/standardized_config.json](data/queries/standardized_config.json).

**Extraction.** `--mode` selects the input regime contrasted in the paper: `global` feeds all papers into a single long-context prompt; `per-paper` runs map-reduce (extract per paper, then aggregate). Documents are read from `mineru_output/{domain}/` by default; pass `--docs-dir` to override.

```bash
export OPENAI_API_KEY=...
python scripts/run_extraction.py \
    --domain social \
    --mode global \
    --model gpt-5.1
```

**Evaluation.** `--semantic-review` invokes an LLM to adjudicate unmatched pairs with moderate similarity (0.3–0.95).

```bash
python scripts/run_evaluation.py \
    --predictions results/social_gpt_5_1_global/predictions.json \
    --ground-truth data/ground_truth/social.json \
    --semantic-review
```

## Citation
```bibtex
@inproceedings{tan2026diagnosing,
  title     = {Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis},
  author    = {Tan, Zhiyin and D'Souza, Jennifer},
  booktitle = {Proceedings of the 22nd Conference on Information and Research Science Connecting to Digital and Library Science (IRCDL 2026)},
  address   = {Modena, Italy},
  month     = feb,
  year      = {2026},
  eprint    = {2602.10881},
  archivePrefix = {arXiv}
}
```

## License

See [LICENSE](LICENSE).

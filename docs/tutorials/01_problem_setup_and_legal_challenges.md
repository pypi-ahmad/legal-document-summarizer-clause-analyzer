# Tutorial 01: Problem Setup and Legal Challenges

## Learning Goals

By the end of this chapter, you should understand:

- why legal summarization and clause analysis are hard
- why domain-specific adaptation matters
- where this intent is encoded in the current repository

## What Is This Technique?

### Definition

Problem framing for legal AI means converting a broad goal ("summarize legal text") into explicit, testable capabilities with a stable output schema.

### Why It Is Used

Without strong framing, legal models drift into generic text generation and fail at operational needs (risk labels, obligations, liabilities, red flags).

### How It Appears in the Code

- output schema and contract: `src/legal_clause_analyzer/schemas.py`
- high-level architecture: `docs/ARCHITECTURE.md`
- end-to-end pipeline steps: `scripts/run_all.py`
- strict completion checks: `scripts/validate_strict_run.py`

### Practical Explanation in This Project

This project defines a single schema (`LegalAnalysis`) and makes every stage (data, baselines, training, evaluation, inference) produce or consume that schema. That is the core reason metrics and artifacts are comparable across systems.

## Why Legal Summarization Is Difficult

- long-range dependencies across clauses
- nested exceptions and carve-outs
- jurisdiction-specific legal language
- ambiguous terms requiring context
- high cost of hallucinated obligations/liabilities

These constraints are reflected in the pipeline design:

- deterministic supervision mapping (for consistency)
- strict schema parsing (for structure)
- multi-axis evaluation (for realism)
- hallucination diagnostics (for safety)

## General LLMs vs Domain-Tuned Legal Models

### General models (prompt-only)

- easy to start
- often inconsistent in legal structure extraction
- weak repeatability for legal risk workflows

### Domain-tuned models (this project)

- higher structure adherence
- improved risk scoring metrics in this run
- still limited on liability recall in current data/training setup

## Project Objective as Implemented

Inputs:

- legal clauses, legal opinions, regulations, legal text passages

Outputs:

- structured legal analysis fields in `LegalAnalysis`

Validation approach:

- baseline-first benchmarking
- fine-tuned comparison
- judge scoring and hallucination diagnostics
- strict-run validation

## Real-Run Context You Should Keep in Mind

From `artifacts/metrics/system_metrics.json` and related reports:

- summary and risk metrics improved after fine-tuning
- obligation/liability recall remains a major gap

This chapter establishes why both can be true at the same time.

## Key Takeaway

The project is not a generic summarizer demo. It is a schema-constrained legal analysis pipeline with measurable behavior and explicit known limitations.

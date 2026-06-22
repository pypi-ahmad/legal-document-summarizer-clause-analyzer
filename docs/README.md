# Documentation Index

This directory contains the complete zero-to-hero documentation set for the legal summarizer and clause analyzer project.

## Start Here

1. Project entry point: `../README.md`
2. Full handbook: `HANDBOOK.md`
3. Chapter tutorials: `tutorials/`
4. PDF bundle: `documentation.pdf`

## Tutorial Chapters

1. `tutorials/01_problem_setup_and_legal_challenges.md`
2. `tutorials/02_dataset_and_supervision_design.md`
3. `tutorials/03_baselines_and_prompting_systems.md`
4. `tutorials/04_finetuning_with_trl_peft_unsloth.md`
5. `tutorials/05_evaluation_and_llm_judging.md`
6. `tutorials/06_error_analysis_and_visualization.md`
7. `tutorials/07_inference_workflow_and_examples.md`
8. `tutorials/08_strict_run_validation_and_artifact_audit.md`

## Source-of-Truth Policy

All quantitative claims in this docs set are grounded in existing project artifacts only:

- `../data/processed/dataset_report.json`
- `../artifacts/metrics/system_metrics.json`
- `../artifacts/metrics/scoreboard.csv`
- `../artifacts/metrics/judge_report.json`
- `../artifacts/metrics/latency_report.json`
- `../artifacts/metrics/hallucination_report.json`
- `../artifacts/metrics/training_runtime_report.json`
- `../artifacts/models/finetuned_adapter/training_backend_report.json`
- `../artifacts/inference/inference_examples.json`

## Navigation Tip

If you are new, follow the chapters in order. If you are implementing in production, start with chapters 4, 5, 7, and 8.

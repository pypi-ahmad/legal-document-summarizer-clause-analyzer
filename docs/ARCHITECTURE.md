# Architecture

## Core Flow

1. Load real LexGLUE subsets with split integrity.
2. Normalize text and labels into a unified sample shape.
3. Convert each row into deterministic structured legal-analysis targets.
4. Run baseline generation systems.
5. Fine-tune base model with TRL + PEFT QLoRA (optional Unsloth acceleration).
6. Evaluate baseline vs fine-tuned with metrics + judges + latency.
7. Run inference workflow and persist structured outputs.

## Components

- `data/lexglue_loader.py`: source-of-truth ingestion
- `data/policy_mapping.py`: deterministic gold-label policy logic
- `data/instruction_builder.py`: instruction-response construction
- `baselines/*`: prompting and model clients
- `finetune/*`: TRL/PEFT training and adapter inference
- `evaluation/*`: metrics, judge, hallucination, plots
- `inference/pipeline.py`: reusable legal-analysis runtime API

## Safety and Reliability Notes

- No train/validation/test leakage
- Seeds set at all pipeline stages
- Structured output schema validated by Pydantic
- Hallucination metrics included in final report
- Runtime backend report documents whether Unsloth or TRL/PEFT path was used
- Explicit legal non-advice disclaimer in docs

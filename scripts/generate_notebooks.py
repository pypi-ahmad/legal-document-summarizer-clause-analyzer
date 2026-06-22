"""Generate tutorial notebooks for the legal clause analyzer project.

The notebooks are generated from code so that tutorial structure stays consistent,
versionable, and reproducible across runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2))


COMMON_SETUP = code_cell(
    "import json\n"
    "import subprocess\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "cwd = Path.cwd().resolve()\n"
    "candidates = [cwd, cwd.parent]\n"
    "ROOT = next((p for p in candidates if (p / 'scripts').exists() and (p / 'configs').exists()), cwd)\n"
    "print('Project root:', ROOT)\n"
    "\n"
    "def run_py(script: str, *args: str) -> None:\n"
    "    cmd = [sys.executable, str(ROOT / script), *args]\n"
    "    print('Running:', ' '.join(cmd))\n"
    "    subprocess.run(cmd, check=True, cwd=str(ROOT))\n"
)


def _technique_block(name: str, details: dict[str, str]) -> str:
    """Return a markdown section with standardized tutorial headings."""

    return (
        f"## What Is This Technique? - {name}\n\n"
        f"### Definition and Core Concepts\n{details['definition']}\n\n"
        f"### Why Was This Technique Developed?\n{details['why_developed']}\n\n"
        f"### What Limitations of Traditional RAG Does It Solve?\n{details['rag_limitations']}\n\n"
        f"### Architecture and Workflow Diagram Explanation\n{details['architecture']}\n\n"
        f"### Component-by-Component Breakdown\n{details['components']}\n\n"
        f"### When Should It Be Used in Real-World Systems?\n{details['when_to_use']}\n\n"
        f"### Advantages and Disadvantages\n"
        f"Advantages:\n{details['advantages']}\n\n"
        f"Disadvantages:\n{details['disadvantages']}\n\n"
        f"### Comparison Against Standard RAG and Other Implemented Variants\n{details['comparison']}\n\n"
        f"### Implementation Details and Design Decisions Used in This Project\n"
        f"{details['implementation_details']}\n"
    )


def build() -> None:
    nb_dir = Path("notebooks")

    n1 = [
        md_cell(
            "# Notebook 01 - Problem Setup, Dataset, and Legal AI Foundations\n\n"
            "This notebook is a zero-to-hero entry point: it explains the problem, the legal data, "
            "and why domain-specific fine-tuning is justified."
        ),
        md_cell(
            "## Why Legal Summarization Is Difficult\n\n"
            "1. Long-context dependencies can spread obligations and exceptions across distant sections.\n"
            "2. Legal terminology has specialized meanings that differ from plain language.\n"
            "3. Ambiguity and nested clauses can invert obligations based on exceptions.\n"
            "4. Cross-references and jurisdiction-specific wording change enforceability.\n"
            "5. Hallucinations can create legal risk by inventing obligations or liabilities."
        ),
        md_cell(
            "## General-Purpose LLMs vs Domain-Specific Legal Fine-Tuning\n\n"
            "General-purpose models can produce fluent summaries but often miss legal structure consistency. "
            "Domain-specific fine-tuning improves schema adherence, extraction reliability, and risk signaling "
            "for repeated legal workflows."
        ),
        md_cell(
            _technique_block(
                "Dataset-Grounded Legal Task Framing",
                {
                    "definition": (
                        "We frame the task as structured legal analysis grounded in real LexGLUE legal datasets, "
                        "not synthetic prompt demos."
                    ),
                    "why_developed": (
                        "Prompt-only systems are easy to start but often inconsistent. A dataset-grounded setup is "
                        "needed for objective baselines and measurable improvement."
                    ),
                    "rag_limitations": (
                        "Traditional RAG improves grounding at inference time but does not directly teach a model how "
                        "to emit a stable legal-analysis schema. This project addresses output-structure learning via "
                        "supervised fine-tuning."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[LexGLUE datasets] --> B[Normalization + split integrity]\n"
                        "B --> C[Policy mapping to structured legal targets]\n"
                        "C --> D[Baselines]\n"
                        "C --> E[Fine-tuning]\n"
                        "D --> F[Evaluation + LLM judges]\n"
                        "E --> F\n"
                        "F --> G[Inference pipeline]\n"
                        "```"
                    ),
                    "components": (
                        "1. Dataset ingestion preserving train/validation/test boundaries.\n"
                        "2. Deterministic policy mapping from legal labels to structured outputs.\n"
                        "3. Baseline systems for prompt-only and zero-shot variants.\n"
                        "4. Fine-tuning stage with parameter-efficient adaptation.\n"
                        "5. Evaluation with metrics, judges, hallucination analysis, and latency."
                    ),
                    "when_to_use": (
                        "Use this setup when you need repeatable legal analysis outputs and auditable model progress."
                    ),
                    "advantages": (
                        "- Reproducible and measurable.\n"
                        "- Works fully local.\n"
                        "- Suitable as a reusable legal AI component."
                    ),
                    "disadvantages": (
                        "- Requires data engineering effort.\n"
                        "- Gold labels are indirect for some target fields (risk/liability).\n"
                        "- Small models still have long-context limits."
                    ),
                    "comparison": (
                        "Compared to standard RAG alone, this method teaches output structure through training. "
                        "Compared to prompt-only baselines, it is more expensive initially but more consistent."
                    ),
                    "implementation_details": (
                        "We use LexGLUE subsets: LEDGAR, UNFAIR-ToS, SCOTUS, EURLEX, CaseHOLD. "
                        "Targets are generated deterministically with explicit policy rules to remain auditable."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "report_path = ROOT / 'data/processed/dataset_report.json'\n"
            "if not report_path.exists():\n"
            "    run_py('scripts/prepare_data.py', '--config', 'configs/default.yaml')\n"
            "report = json.loads(report_path.read_text())\n"
            "report"
        ),
        md_cell(
            "## Dataset Structure and Labeling Schema\n\n"
            "- Mix of single-label and multi-label legal tasks.\n"
            "- Metadata includes dataset identity, split, and legal label names.\n"
            "- Regulatory hierarchy represented through mixed domains: contract law, US case law, EU law."
        ),
        code_cell(
            "fig_path = ROOT / 'artifacts/figures/dataset_distributions.png'\n"
            "img = plt.imread(fig_path)\n"
            "plt.figure(figsize=(12, 4))\n"
            "plt.imshow(img)\n"
            "plt.axis('off')\n"
            "plt.title('Dataset and Risk Distributions')\n"
            "plt.show()"
        ),
        md_cell(
            "## Dataset Limitations\n\n"
            "1. No native executive-summary labels across all subsets.\n"
            "2. No direct gold risk/liability taxonomy for our final schema.\n"
            "3. Cross-jurisdiction semantic differences remain challenging for compact models.\n\n"
            "These limitations are handled with deterministic policy mapping and explicit evaluation."
        ),
    ]

    n2 = [
        md_cell(
            "# Notebook 02 - Baseline Systems Before Fine-Tuning\n\n"
            "This notebook implements and benchmarks baseline legal analysis systems first."
        ),
        md_cell(
            _technique_block(
                "Prompt-Only and Few-Shot Baseline Benchmarking",
                {
                    "definition": (
                        "Baseline benchmarking measures what non-fine-tuned systems can do before any adaptation."
                    ),
                    "why_developed": (
                        "Without baselines, post-training results are not interpretable."
                    ),
                    "rag_limitations": (
                        "RAG can improve factual grounding, but it does not guarantee legal-schema compliance. "
                        "Baseline prompting reveals this gap explicitly."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[Test legal rows] --> B[Prompt-only]\n"
                        "A --> C[Few-shot]\n"
                        "A --> D[Granite zero-shot]\n"
                        "A --> E[Qwen zero-shot]\n"
                        "B --> F[Predictions]\n"
                        "C --> F\n"
                        "D --> F\n"
                        "E --> F\n"
                        "```"
                    ),
                    "components": (
                        "1. Prompt templates with schema constraints.\n"
                        "2. Few-shot exemplars.\n"
                        "3. Two base local models.\n"
                        "4. Structured parser and fallback handling."
                    ),
                    "when_to_use": (
                        "Always use before fine-tuning to establish a real lower bound."
                    ),
                    "advantages": (
                        "- Low setup cost.\n"
                        "- Fast iteration.\n"
                        "- Useful diagnostic reference."
                    ),
                    "disadvantages": (
                        "- Output inconsistency.\n"
                        "- Higher schema-break probability.\n"
                        "- Weaker extraction fidelity."
                    ),
                    "comparison": (
                        "Compared with standard RAG baselines, this benchmark isolates generation behavior without "
                        "retrieval confounders. It gives a clean before/after fine-tuning comparison."
                    ),
                    "implementation_details": (
                        "We run four systems and persist `predictions_*.jsonl` files with latency fields for later "
                        "analysis."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "run_dir = ROOT / 'artifacts/runs/balanced_local_v1'\n"
            "if not list(run_dir.glob('predictions_*.jsonl')):\n"
            "    run_py('scripts/run_baselines.py', '--config', 'configs/default.yaml')\n"
            "sorted([p.name for p in run_dir.glob('predictions_*.jsonl')])"
        ),
        code_cell(
            "samples = []\n"
            "for path in sorted(run_dir.glob('predictions_*.jsonl')):\n"
            "    with path.open() as f:\n"
            "        first = json.loads(next(f))\n"
            "        samples.append({\n"
            "            'system': path.stem.replace('predictions_', ''),\n"
            "            'summary': first['prediction']['executive_summary'][:180],\n"
            "            'risk_level': first['prediction']['risk_level'],\n"
            "            'latency_seconds': first.get('latency_seconds', 0.0),\n"
            "        })\n"
            "pd.DataFrame(samples)"
        ),
        md_cell(
            "## Baseline Observations\n\n"
            "Use these baseline outputs as the direct comparator when judging fine-tuned improvements in "
            "quality, schema consistency, and latency."
        ),
    ]

    n3 = [
        md_cell(
            "# Notebook 03 - Data Preparation, Policy Mapping, and Long-Text Handling\n\n"
            "This notebook explains every transformation from raw legal rows to instruction-response supervision."
        ),
        md_cell(
            _technique_block(
                "Deterministic Policy Mapping for Legal Supervision",
                {
                    "definition": (
                        "Policy mapping converts existing legal labels and text cues into the project output schema "
                        "without synthetic LLM-generated labels."
                    ),
                    "why_developed": (
                        "LexGLUE labels are task-specific; we need one unified schema for summarization, risk, "
                        "liability, and clause extraction."
                    ),
                    "rag_limitations": (
                        "RAG retrieves context but does not produce structured supervision targets for training. "
                        "Policy mapping closes that training-data gap."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[Raw legal text + gold labels] --> B[Normalization]\n"
                        "B --> C[Risk mapping]\n"
                        "B --> D[Liability mapping]\n"
                        "B --> E[Obligation/right/restriction extraction]\n"
                        "C --> F[Structured LegalAnalysis target]\n"
                        "D --> F\n"
                        "E --> F\n"
                        "```"
                    ),
                    "components": (
                        "1. Text cleaning and truncation.\n"
                        "2. Keyword-driven risk and liability mapping.\n"
                        "3. Clause cue extraction for obligations/rights/restrictions.\n"
                        "4. Evidence span capture for hallucination auditing.\n"
                        "5. Instruction + input + JSON output formatting."
                    ),
                    "when_to_use": (
                        "Use when you have real domain labels but not a ready-made instruction-tuning target format."
                    ),
                    "advantages": (
                        "- Auditable and deterministic.\n"
                        "- No synthetic labeling dependence.\n"
                        "- Easy to maintain and debug."
                    ),
                    "disadvantages": (
                        "- Can miss nuanced semantics.\n"
                        "- Rule updates needed for new jurisdictions.\n"
                        "- Less expressive than expert-annotated legal analyses."
                    ),
                    "comparison": (
                        "Compared with standard RAG pipelines, this approach focuses on train-time schema "
                        "construction rather than inference-time retrieval."
                    ),
                    "implementation_details": (
                        "Policy rules live in `configs/policy_mapping.yaml`; transformations are implemented in "
                        "`data/policy_mapping.py` and `data/instruction_builder.py`."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "train_path = ROOT / 'data/processed/train_supervised.jsonl'\n"
            "if not train_path.exists():\n"
            "    run_py('scripts/prepare_data.py', '--config', 'configs/default.yaml')\n"
            "preview = []\n"
            "with train_path.open() as f:\n"
            "    for i, line in enumerate(f):\n"
            "        if i >= 2:\n"
            "            break\n"
            "        preview.append(json.loads(line))\n"
            "preview"
        ),
        code_cell(
            "policy = json.loads((ROOT / 'data/processed/dataset_report.json').read_text())\n"
            "pd.DataFrame([\n"
            "    {'split': k, 'rows': v} for k, v in policy['row_counts'].items()\n"
            "])"
        ),
        md_cell(
            "## Long-Document Handling Strategy\n\n"
            "Current approach uses conservative truncation for local training constraints. "
            "For production, add chunking + sliding-window map-reduce summarization for very long contracts."
        ),
    ]

    n4 = [
        md_cell(
            "# Notebook 04 - Fine-Tuning with TRL, PEFT, and Optional Unsloth\n\n"
            "This chapter explains why TRL and PEFT are required, and where Unsloth is conditionally used."
        ),
        md_cell(
            "## Tool Definitions and Why They Matter\n\n"
            "### TRL\n"
            "TRL (Transformer Reinforcement Learning) provides modern supervised fine-tuning trainers (`SFTTrainer`) "
            "with LLM-specific conveniences.\n\n"
            "### PEFT\n"
            "PEFT enables parameter-efficient tuning (LoRA/QLoRA), reducing trainable parameters and VRAM demand.\n\n"
            "### Unsloth\n"
            "Unsloth is an optimization framework that can accelerate LoRA/QLoRA training and inference on supported "
            "models. In this project it is optional, not forced."
        ),
        md_cell(
            "## Official Documentation Review Used Before Implementation (Checked: 2026-06-21)\n\n"
            "- TRL official docs (`/huggingface/trl`): validated `SFTTrainer` + `SFTConfig` usage for supervised "
            "instruction tuning and QLoRA.\n"
            "- PEFT official docs (`/huggingface/peft`): validated `prepare_model_for_kbit_training` before "
            "`get_peft_model`, and LoRA target module configuration.\n"
            "- Unsloth official docs (`/unslothai/unsloth`): validated optional backend path using "
            "`FastLanguageModel.from_pretrained` and `FastLanguageModel.get_peft_model`.\n\n"
            "These references informed the exact order of operations in `src/legal_clause_analyzer/finetune/train.py`."
        ),
        md_cell(
            _technique_block(
                "QLoRA Fine-Tuning for Legal Structured Generation",
                {
                    "definition": (
                        "QLoRA combines 4-bit quantization with LoRA adapters so we can fine-tune local 3B/4B models "
                        "on consumer GPUs."
                    ),
                    "why_developed": (
                        "Full fine-tuning is expensive and often impractical on local hardware."
                    ),
                    "rag_limitations": (
                        "RAG improves context access but does not adapt generation behavior itself. QLoRA teaches the "
                        "model to emit legal-analysis structure directly."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[Instruction dataset] --> B[TRL SFTTrainer]\n"
                        "B --> C[PEFT LoRA adapters]\n"
                        "C --> D[4-bit quantized base model]\n"
                        "D --> E[Fine-tuned legal analyzer adapter]\n"
                        "```\n\n"
                        "If Unsloth is installed and compatible, model loading/adapter wiring uses Unsloth APIs first; "
                        "otherwise the pipeline falls back to TRL+PEFT."
                    ),
                    "components": (
                        "1. Quantized model loading.\n"
                        "2. LoRA configuration and adapter attachment.\n"
                        "3. TRL SFT trainer loop.\n"
                        "4. Backend auto-selection (Unsloth or TRL+PEFT).\n"
                        "5. Runtime artifact logging for memory and backend choice."
                    ),
                    "when_to_use": (
                        "Use for domain-specific behavior adaptation when prompt engineering alone is insufficient."
                    ),
                    "advantages": (
                        "- Works on single local GPU.\n"
                        "- Faster iteration than full fine-tuning.\n"
                        "- Better schema consistency."
                    ),
                    "disadvantages": (
                        "- Adapter quality depends on supervision quality.\n"
                        "- Small models still have reasoning limits.\n"
                        "- Backend compatibility varies by model/runtime."
                    ),
                    "comparison": (
                        "Compared to standard RAG, QLoRA changes model behavior persistently. Compared to prompt-only "
                        "baselines, it requires training time but yields more stable structured outputs."
                    ),
                    "implementation_details": (
                        "This project uses TRL `SFTTrainer`, PEFT `prepare_model_for_kbit_training` + `LoraConfig`, "
                        "and optional Unsloth `FastLanguageModel` when available."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "metrics_path = ROOT / 'artifacts/models/finetuned_adapter/training_metrics.json'\n"
            "if not metrics_path.exists():\n"
            "    run_py('scripts/train_model.py', '--config', 'configs/default.yaml')\n"
            "train_metrics = json.loads(metrics_path.read_text())\n"
            "train_metrics"
        ),
        code_cell(
            "backend_report_path = ROOT / 'artifacts/models/finetuned_adapter/training_backend_report.json'\n"
            "backend_report = json.loads(backend_report_path.read_text())\n"
            "backend_report"
        ),
        code_cell(
            "runtime_path = ROOT / 'artifacts/metrics/training_runtime_report.json'\n"
            "runtime_report = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}\n"
            "selected_backend = backend_report.get('selected_backend', 'unknown')\n"
            "tool_coverage = pd.DataFrame([\n"
            "    {\n"
            "        'tool': 'TRL',\n"
            "        'definition': 'Trainer framework for LLM fine-tuning (SFTTrainer).',\n"
            "        'why_used': 'Needed a robust supervised fine-tuning loop and logging.',\n"
            "        'where_used': 'finetune/train.py::_build_sft_config + _build_trainer',\n"
            "        'what_changed': 'Unified training config and checkpoint/eval behavior.',\n"
            "        'observed_effect': 'Training artifacts (loss curves, metrics) produced consistently.',\n"
            "    },\n"
            "    {\n"
            "        'tool': 'PEFT',\n"
            "        'definition': 'Parameter-efficient adaptation via LoRA/QLoRA adapters.',\n"
            "        'why_used': 'Full fine-tuning was too heavy for local 8GB-class GPUs.',\n"
            "        'where_used': 'prepare_model_for_kbit_training + get_peft_model in finetune/train.py',\n"
            "        'what_changed': 'Only adapter parameters are trained.',\n"
            "        'observed_effect': 'Fine-tuned adapter artifacts and measurable metric deltas vs baselines.',\n"
            "    },\n"
            "    {\n"
            "        'tool': 'Unsloth',\n"
            "        'definition': 'Optional accelerated LoRA/QLoRA backend on supported models.',\n"
            "        'why_used': 'Potential speed/VRAM benefit where compatible.',\n"
            "        'where_used': 'finetune/train.py::_load_unsloth_bundle (auto-first fallback path)',\n"
            "        'what_changed': f\"Runtime backend order = {backend_report.get('backend_order', [])}\",\n"
            "        'observed_effect': f\"Selected backend in this run = {selected_backend}\",\n"
            "    },\n"
            "])\n"
            "tool_coverage"
        ),
        code_cell(
            "curve_png = ROOT / 'artifacts/figures/training_curve.png'\n"
            "img = plt.imread(curve_png)\n"
            "plt.figure(figsize=(10, 4))\n"
            "plt.imshow(img)\n"
            "plt.axis('off')\n"
            "plt.title('Training and Validation Curves')\n"
            "plt.show()"
        ),
        md_cell(
            "## What Changed Because of TRL, PEFT, and Unsloth?\n\n"
            "1. **TRL** changed the project from ad-hoc loops to a reproducible SFT pipeline with consistent "
            "training/eval/checkpoint artifacts.\n"
            "2. **PEFT** changed the optimization target from full-model updates to adapter-only updates, enabling "
            "QLoRA on local hardware budgets.\n"
            "3. **Unsloth** changed backend selection strategy: it is attempted first only when available and useful, "
            "then cleanly falls back to TRL+PEFT.\n\n"
            "Use the `tool_coverage` table and backend/runtime artifacts above to connect tool choice to observed "
            "post-run behavior in your environment."
        ),
    ]

    n5 = [
        md_cell(
            "# Notebook 05 - Evaluation, Benchmarking, and LLM-as-a-Judge\n\n"
            "This notebook computes and interprets quantitative and judge-based metrics."
        ),
        md_cell(
            _technique_block(
                "Multi-Dimensional Legal Evaluation",
                {
                    "definition": (
                        "Evaluation combines lexical similarity, classification quality, extraction F1, hallucination "
                        "rates, latency, and LLM-judge scoring."
                    ),
                    "why_developed": (
                        "No single metric can capture legal output quality and risk."
                    ),
                    "rag_limitations": (
                        "RAG-centric retrieval metrics (Recall@k/MRR) do not measure legal output schema quality. "
                        "This evaluation fills that generation-quality gap."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[Prediction files] --> B[Metric engine]\n"
                        "A --> C[Hallucination analysis]\n"
                        "A --> D[LLM judges]\n"
                        "B --> E[CSV/JSON reports]\n"
                        "C --> E\n"
                        "D --> E\n"
                        "E --> F[Plots + interpretation]\n"
                        "```"
                    ),
                    "components": (
                        "1. Summarization: ROUGE, BLEU, METEOR, BERTScore.\n"
                        "2. Risk classification: accuracy/precision/recall/F1.\n"
                        "3. Clause extraction: precision/recall/F1.\n"
                        "4. Hallucination diagnostics.\n"
                        "5. LLM-as-a-judge using Granite and Qwen."
                    ),
                    "when_to_use": (
                        "Use in any production-bound legal AI workflow where reliability matters."
                    ),
                    "advantages": (
                        "- Captures multiple failure modes.\n"
                        "- Supports baseline-vs-finetuned comparison.\n"
                        "- Provides operational latency signals."
                    ),
                    "disadvantages": (
                        "- Judge scores are model-dependent.\n"
                        "- Lexical metrics may miss legal nuance.\n"
                        "- More expensive than single-metric evaluation."
                    ),
                    "comparison": (
                        "Compared with standard RAG evaluations focused on retrieval relevance, this notebook measures "
                        "downstream legal generation quality and structure."
                    ),
                    "implementation_details": (
                        "Evaluation artifacts are written to `artifacts/metrics/` and `artifacts/figures/`."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "metrics_json = ROOT / 'artifacts/metrics/system_metrics.json'\n"
            "if not metrics_json.exists():\n"
            "    run_py('scripts/evaluate.py', '--config', 'configs/default.yaml')\n"
            "metrics = json.loads(metrics_json.read_text())\n"
            "metrics.keys()"
        ),
        code_cell(
            "metrics_df = pd.read_csv(ROOT / 'artifacts/metrics/system_metrics.csv')\n"
            "metrics_df.head(20)"
        ),
        code_cell(
            "scoreboard_path = ROOT / 'artifacts/metrics/scoreboard.csv'\n"
            "scoreboard = pd.read_csv(scoreboard_path) if scoreboard_path.exists() else pd.DataFrame()\n"
            "scoreboard"
        ),
        code_cell(
            "latency = json.loads((ROOT / 'artifacts/metrics/latency_report.json').read_text())\n"
            "pd.DataFrame(latency).T"
        ),
        code_cell(
            "for name in ['metric_bars.png', 'risk_confusion_matrix.png', 'system_scoreboard.png']:\n"
            "    p = ROOT / 'artifacts/figures' / name\n"
            "    if p.exists():\n"
            "        plt.figure(figsize=(10, 4))\n"
            "        plt.imshow(plt.imread(p))\n"
            "        plt.axis('off')\n"
            "        plt.title(name)\n"
            "        plt.show()"
        ),
        code_cell(
            "hall = json.loads((ROOT / 'artifacts/metrics/hallucination_report.json').read_text())\n"
            "judge = json.loads((ROOT / 'artifacts/metrics/judge_report.json').read_text())\n"
            "pivot = scoreboard.pivot_table(index='system', columns='metric', values='score', aggfunc='mean') if not scoreboard.empty else pd.DataFrame()\n"
            "best_metrics = pivot.idxmax().to_dict() if not pivot.empty else {}\n"
            "deltas = {}\n"
            "if not pivot.empty and 'fine_tuned_adapter' in pivot.index:\n"
            "    for metric in pivot.columns:\n"
            "        others = pivot.loc[pivot.index != 'fine_tuned_adapter', metric].dropna()\n"
            "        if not others.empty:\n"
            "            deltas[metric] = float(pivot.loc['fine_tuned_adapter', metric] - others.max())\n"
            "\n"
            "summary = {\n"
            "    'best_system_per_metric': best_metrics,\n"
            "    'fine_tuned_delta_vs_best_baseline': deltas,\n"
            "    'hallucination_report': hall,\n"
            "    'judge_report': judge,\n"
            "}\n"
            "summary"
        ),
        md_cell(
            "## Post-Run Interpretation and Lessons\n\n"
            "1. Read `best_system_per_metric` and `fine_tuned_delta_vs_best_baseline` first to quantify where "
            "fine-tuning helped or did not help.\n"
            "2. Cross-check numeric gains against `hallucination_report` and `judge_report` to detect quality vs "
            "faithfulness tradeoffs.\n"
            "3. Use `latency_report` to decide whether the accuracy gain justifies operational cost.\n"
            "4. If metrics regress or stall, inspect Notebook 06 row-level errors before retraining."
        ),
    ]

    n6 = [
        md_cell(
            "# Notebook 06 - Error Analysis, Failure Modes, and Lessons Learned\n\n"
            "This notebook turns metric deltas into actionable model-improvement hypotheses."
        ),
        md_cell(
            _technique_block(
                "Structured Legal Error Analysis",
                {
                    "definition": (
                        "Error analysis inspects concrete examples to explain misses, hallucinations, and "
                        "misclassifications."
                    ),
                    "why_developed": (
                        "Aggregate metrics do not reveal root causes."
                    ),
                    "rag_limitations": (
                        "RAG may reduce unsupported claims by grounding, but legal extraction errors still occur if "
                        "the generator is not specialized."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[Predictions + references] --> B[Error slicing]\n"
                        "B --> C[Missed risks]\n"
                        "B --> D[Hallucinated obligations]\n"
                        "B --> E[Liability extraction failures]\n"
                        "C --> F[Actionable remediations]\n"
                        "D --> F\n"
                        "E --> F\n"
                        "```"
                    ),
                    "components": (
                        "1. Hallucination rate report.\n"
                        "2. Confusion matrix inspection.\n"
                        "3. Row-level prediction/reference comparison.\n"
                        "4. Error taxonomy and remediation ideas."
                    ),
                    "when_to_use": (
                        "Use after every major model iteration before production promotion."
                    ),
                    "advantages": (
                        "- Surfaces concrete failure patterns.\n"
                        "- Improves data and prompt decisions.\n"
                        "- Supports risk-focused legal QA."
                    ),
                    "disadvantages": (
                        "- Requires manual review effort.\n"
                        "- Can be biased by small slices if sample size is low."
                    ),
                    "comparison": (
                        "Compared with standard RAG debugging, this notebook focuses on output schema and legal "
                        "decision quality rather than retrieval-only tuning."
                    ),
                    "implementation_details": (
                        "Error artifacts and visualizations are generated in the evaluation stage and inspected here."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "run_dir = ROOT / 'artifacts/runs/balanced_local_v1'\n"
            "pred_file = run_dir / 'predictions_fine_tuned_adapter.jsonl'\n"
            "if not pred_file.exists():\n"
            "    pred_file = run_dir / 'predictions_qwen_zero_shot.jsonl'\n"
            "\n"
            "rows = []\n"
            "with pred_file.open() as f:\n"
            "    for line in f:\n"
            "        rows.append(json.loads(line))\n"
            "len(rows), pred_file.name"
        ),
        code_cell(
            "hall = json.loads((ROOT / 'artifacts/metrics/hallucination_report.json').read_text())\n"
            "pd.DataFrame(hall).T"
        ),
        code_cell(
            "records = []\n"
            "for row in rows[:25]:\n"
            "    records.append({\n"
            "        'row_id': row['row_id'],\n"
            "        'dataset': row['dataset'],\n"
            "        'ref_risk': row['reference']['risk_level'],\n"
            "        'pred_risk': row['prediction']['risk_level'],\n"
            "        'ref_liabilities': ', '.join(row['reference'].get('liabilities', [])),\n"
            "        'pred_liabilities': ', '.join(row['prediction'].get('liabilities', [])),\n"
            "    })\n"
            "pd.DataFrame(records).head(10)"
        ),
        code_cell(
            "for name in ['risk_confusion_matrix.png', 'metric_bars.png', 'system_scoreboard.png']:\n"
            "    p = ROOT / 'artifacts/figures' / name\n"
            "    if p.exists():\n"
            "        plt.figure(figsize=(10, 4))\n"
            "        plt.imshow(plt.imread(p))\n"
            "        plt.axis('off')\n"
            "        plt.title(name)\n"
            "        plt.show()"
        ),
        md_cell(
            "## Lessons Learned and Practical Takeaways\n\n"
            "1. Review row-level errors, not only aggregate metrics.\n"
            "2. Fix recurring misses with targeted data improvements or adapter retraining.\n"
            "3. Keep hallucination checks as a mandatory legal safety gate."
        ),
    ]

    n7 = [
        md_cell(
            "# Notebook 07 - Inference Workflow and Production Usage\n\n"
            "Final notebook for practical inference, output inspection, and deployment readiness."
        ),
        md_cell(
            _technique_block(
                "Structured Legal Inference Pipeline",
                {
                    "definition": (
                        "A production inference path that prioritizes fine-tuned adapters and falls back safely when "
                        "adapter loading fails."
                    ),
                    "why_developed": (
                        "Production systems need resilient behavior, explicit output schema, and predictable latency."
                    ),
                    "rag_limitations": (
                        "RAG can feed better context but still needs a robust generator and output contract. This "
                        "pipeline focuses on deterministic legal output structure."
                    ),
                    "architecture": (
                        "```mermaid\n"
                        "flowchart LR\n"
                        "A[Input legal text] --> B{Adapter available?}\n"
                        "B -->|Yes| C[Fine-tuned adapter inference]\n"
                        "B -->|No| D[Baseline generation fallback]\n"
                        "C --> E[Schema validation]\n"
                        "D --> E\n"
                        "E --> F[Executive summary + obligations + liabilities + risk]\n"
                        "```"
                    ),
                    "components": (
                        "1. Adapter-first inference selection.\n"
                        "2. Baseline fallback path.\n"
                        "3. Structured parser and schema validation.\n"
                        "4. Latency capture and persisted examples."
                    ),
                    "when_to_use": (
                        "Use in legal research assistants, contract review tools, compliance workflows, and "
                        "regulatory intelligence systems."
                    ),
                    "advantages": (
                        "- Robust fallback behavior.\n"
                        "- Structured outputs ready for downstream systems.\n"
                        "- Supports local-only deployment."
                    ),
                    "disadvantages": (
                        "- Small models can still miss subtle legal nuance.\n"
                        "- Latency/quality tradeoffs depend on model choice and hardware."
                    ),
                    "comparison": (
                        "Compared with standard RAG-only responses, this output contract is directly consumable by "
                        "policy engines, dashboards, and legal ops tooling."
                    ),
                    "implementation_details": (
                        "Inference now records latency and indicates strategy (`fine_tuned_adapter` or fallback)."
                    ),
                },
            )
        ),
        COMMON_SETUP,
        code_cell(
            "inference_path = ROOT / 'artifacts/inference/inference_examples.json'\n"
            "if not inference_path.exists():\n"
            "    run_py('scripts/run_inference.py', '--config', 'configs/default.yaml')\n"
            "examples = json.loads(inference_path.read_text())\n"
            "pd.DataFrame([\n"
            "    {\n"
            "        'example_id': x['example_id'],\n"
            "        'strategy': x['strategy'],\n"
            "        'model_used': x['model_used'],\n"
            "        'risk_level': x['analysis']['risk_level'],\n"
            "        'latency_seconds': x.get('latency_seconds', 0.0),\n"
            "    }\n"
            "    for x in examples\n"
            "])"
        ),
        code_cell(
            "examples[0]"
        ),
        code_cell(
            "summary = {\n"
            "    'strategy_counts': pd.Series([x['strategy'] for x in examples]).value_counts().to_dict(),\n"
            "    'mean_latency_seconds': float(np.mean([x.get('latency_seconds', 0.0) for x in examples])),\n"
            "    'max_latency_seconds': float(np.max([x.get('latency_seconds', 0.0) for x in examples])),\n"
            "}\n"
            "summary"
        ),
        md_cell(
            "## Post-Run Output Analysis\n\n"
            "Interpret outputs with the following checklist:\n"
            "1. Is the executive summary legally faithful and plain-English?\n"
            "2. Are obligations/rights/liabilities supported by source wording?\n"
            "3. Does risk classification include a clear rationale?\n"
            "4. Are red flags actionable for legal review teams?"
        ),
        md_cell(
            "## Final Conclusion (Measured)\n\n"
            "Use `summary`, `system_metrics.csv`, `scoreboard.csv`, `hallucination_report.json`, and "
            "`latency_report.json` to produce a run-specific conclusion. A strong conclusion should state:\n"
            "- Which system won each key metric and by how much.\n"
            "- Whether hallucination behavior improved or regressed.\n"
            "- Whether current latency is acceptable for legal operations.\n"
            "- Which concrete data/model changes are next for production hardening."
        ),
    ]

    write_notebook(nb_dir / "01_problem_setup_dataset.ipynb", n1)
    write_notebook(nb_dir / "02_baselines_prompting.ipynb", n2)
    write_notebook(nb_dir / "03_data_preparation_policy_mapping.ipynb", n3)
    write_notebook(nb_dir / "04_finetuning_qlora.ipynb", n4)
    write_notebook(nb_dir / "05_evaluation_and_judging.ipynb", n5)
    write_notebook(nb_dir / "06_error_analysis_and_visualization.ipynb", n6)
    write_notebook(nb_dir / "07_inference_workflow.ipynb", n7)


if __name__ == "__main__":
    build()

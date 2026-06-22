"""Inference with fine-tuned LoRA adapter over the base causal model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from legal_clause_analyzer.baselines.generation import safe_analysis_from_text
from legal_clause_analyzer.finetune.dataset_format import format_for_sft
from legal_clause_analyzer.schemas import LegalAnalysis


def _prompt_for_inference(text: str) -> str:
    """Prompt template aligned with training formatting."""

    return (
        "### Instruction\n"
        "Analyze this legal text and return structured JSON with executive summary, obligations, rights, "
        "restrictions, liabilities, risk level, red flags, recommendations, and evidence spans.\n\n"
        "### Input\n"
        f"{text}\n\n"
        "### Output JSON\n"
    )


def load_finetuned_model(
    base_model: str,
    adapter_dir: Path,
    max_memory_gpu_gib: float = 7.0,
    max_memory_cpu_gib: float = 24.0,
    bf16: bool = True,
    use_4bit: bool = True,
):
    """Load quantized base model and attach LoRA adapter."""

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    offload_dir = (adapter_dir / "offload_cache").resolve()
    offload_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quant_cfg,
        max_memory={0: f"{max_memory_gpu_gib:.1f}GiB", "cpu": f"{max_memory_cpu_gib:.1f}GiB"},
        offload_folder=str(offload_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir.resolve()))
    model.eval()
    return model, tokenizer


def generate_with_adapter(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 420,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> LegalAnalysis:
    """Generate one structured prediction from adapter model."""

    prompt = _prompt_for_inference(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return safe_analysis_from_text(generated)

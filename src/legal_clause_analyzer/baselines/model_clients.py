"""Model client abstractions for Ollama and Transformers backends."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class GenerationConfig:
    """Shared generation controls passed to backends."""

    temperature: float
    top_p: float
    max_tokens: int
    num_ctx: int
    request_timeout_seconds: float = 45.0


class OllamaClient:
    """Thin wrapper around Ollama chat API."""

    def __init__(self, model: str):
        self.model = model
        try:
            import ollama
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Ollama Python package is not installed") from exc

        self._ollama = ollama

    def generate(self, prompt: str, config: GenerationConfig, system_prompt: str | None = None) -> str:
        """Generate completion using local Ollama model."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._ollama.Client(timeout=config.request_timeout_seconds)
        response = client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                "num_ctx": config.num_ctx,
            },
        )
        return str(response["message"]["content"])


class TransformersClient:
    """Fallback client that runs Hugging Face causal models directly."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        logger.info("Loading Transformers model for inference: {}", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, prompt: str, config: GenerationConfig, system_prompt: str | None = None) -> str:
        self._load()
        merged_prompt = prompt if not system_prompt else f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\n{prompt}"
        outputs = self._pipe(
            merged_prompt,
            max_new_tokens=config.max_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            return_full_text=False,
        )
        return str(outputs[0]["generated_text"])


class UnifiedModelClient:
    """Backend-agnostic generation wrapper."""

    def __init__(self, backend: str, model: str):
        if backend == "ollama":
            self._client = OllamaClient(model=model)
        elif backend == "transformers":
            self._client = TransformersClient(model_id=model)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompt: str, config: GenerationConfig, system_prompt: str | None = None) -> str:
        return self._client.generate(prompt=prompt, config=config, system_prompt=system_prompt)


def try_parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from model output."""

    text = text.strip()
    if not text:
        return None

    candidates: list[str] = [text]

    # Common pattern: model wraps JSON inside ```json ... ```.
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend([item.strip() for item in fenced if item.strip()])

    # Try extracting all brace-delimited snippets and parse the longest first.
    brace_snippets = re.findall(r"\{[\s\S]*\}", text)
    brace_snippets = sorted({s.strip() for s in brace_snippets if s.strip()}, key=len, reverse=True)
    candidates.extend(brace_snippets)

    def _attempt_parse(candidate: str) -> dict[str, Any] | None:
        for variant in [candidate, candidate.replace("\n", " ").strip()]:
            try:
                obj = json.loads(variant)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

        # Fallback: tolerate Python-like dicts / single quotes.
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
        except Exception:
            return None
        return None

    for candidate in candidates:
        parsed = _attempt_parse(candidate)
        if parsed is not None:
            return parsed

    logger.debug("Could not parse model output into JSON dict. Preview: {}", text[:200])
    return None

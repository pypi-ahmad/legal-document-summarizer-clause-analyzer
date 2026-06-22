"""Export helper for deploying fine-tuned adapters in Ollama.

Ollama does not train models itself; it imports trained adapters/models via Modelfile.
"""

from __future__ import annotations

from pathlib import Path


def write_modelfile(base_model: str, adapter_dir: Path, output_dir: Path) -> Path:
    """Create Modelfile that binds base model + adapter path.

    Args:
        base_model: Ollama base model tag.
        adapter_dir: Path to fine-tuned adapter weights directory.
        output_dir: Output folder where Modelfile is written.

    Returns:
        Path to generated Modelfile.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    modelfile = output_dir / "Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM {base_model}",
                f"ADAPTER {adapter_dir.resolve()}",
                "PARAMETER temperature 0.1",
                "PARAMETER num_ctx 8192",
                "SYSTEM You are a specialized legal document summarizer and clause analyzer.",
                "",
            ]
        )
    )
    return modelfile

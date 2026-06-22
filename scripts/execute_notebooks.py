"""Execute tutorial notebooks in order using nbconvert."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute notebooks end-to-end")
    parser.add_argument("--notebooks-dir", default="notebooks")
    args = parser.parse_args()

    notebooks = sorted(
        notebook
        for notebook in Path(args.notebooks_dir).glob("*.ipynb")
        if not notebook.name.endswith(".executed.ipynb")
    )
    for notebook in notebooks:
        out_name = notebook.stem + ".executed.ipynb"
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook),
            "--output",
            out_name,
        ]
        logger.info("Executing notebook: {}", notebook.name)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

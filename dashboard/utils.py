import ast
import os
import json
import math
import re
import shutil
import statistics
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import seaborn as sns


def get_pdf_text(file_path):
    doc = pymupdf.open(file_path)
    texts = []
    for page in doc:
        temp = page.get_text()
        texts.append(temp.strip())
    doc.close()
    return texts


def render_pdf_pages(file_path, output_dir, zoom=1.5):
    doc = pymupdf.open(file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    try:
        for index, page in enumerate(doc, start=1):
            image_path = output_path / f"page_{index}.png"
            if not image_path.exists():
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                image_path.write_bytes(pixmap.tobytes("png"))
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths


def render_csv_thumbnail(file_path: str, output_dir: str) -> str:
    """Generate a thumbnail image from a CSV file showing the first few rows."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = output_path / "preview.png"
    
    if image_path.exists():
        return str(image_path)
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, nrows=10)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Display table with limited rows
        table = ax.table(
            cellText=df.head(8).values,
            colLabels=df.columns,
            cellLoc='left',
            loc='center',
            colWidths=[min(20, len(str(col)) * 1.2) / 100 for col in df.columns]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, min(9, len(df) + 1)):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        # Save figure
        plt.tight_layout()
        fig.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return str(image_path)
    except Exception as e:
        # Return an error indicator or empty path
        print(f"Error generating CSV thumbnail: {e}")
        return ""


# These functions have been moved to ``fin_ai.core.history``.
# The code below re-exports them for backward compatibility.
from fin_ai.core.history import (  # noqa: E402, F401
    load_question_history,
    save_question_history,
    append_question_history,
    clear_question_history,
    purge_vector_db_assets,
    get_question_history_path,
)


def extract_python_code(response_text: str) -> str:
    code_block_match = re.search(
        r"```python\s*(.*?)```",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    generic_block_match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if generic_block_match:
        return generic_block_match.group(1).strip()

    return response_text.strip()


def sanitize_generated_python_code(code: str) -> str:
    cleaned_lines = []
    for line in code.splitlines():
        normalized_line = line
        if re.match(r"^\s*(from\s+\S+\s+import\s+.+|import\s+\S+.*)\s+to\s+specify\b", line):
            normalized_line = re.sub(r"\s+to\s+specify\b.*$", "", line)
        cleaned_lines.append(normalized_line)
    return "\n".join(cleaned_lines).strip()


def validate_python_code(code: str) -> None:
    allowed_modules = {
        "math",
        "statistics",
        "json",
        "matplotlib",
        "matplotlib.pyplot",
        "seaborn",
        "pandas",
        "yfinance",
    }
    blocked_names = {
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "__import__",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
    }

    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_modules:
                    raise ValueError(f"Import '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name not in allowed_modules:
                raise ValueError(f"Import from '{module_name}' is not allowed.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_names:
                raise ValueError(f"Function '{node.func.id}' is not allowed.")
        elif isinstance(node, ast.Name):
            if node.id in blocked_names:
                raise ValueError(f"Name '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("Dunder attribute access is not allowed.")


def execute_python_code(code: str) -> tuple[str, str]:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    validate_python_code(code)

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    execution_globals = {
        "__name__": "__main__",
        "__builtins__": safe_builtins,
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "math": math,
        "statistics": statistics,
        "json": json,
    }

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, execution_globals, execution_globals)
    except Exception as exc:
        error_output = stderr_buffer.getvalue()
        if error_output:
            error_output = f"{error_output}\n{exc}"
        else:
            error_output = str(exc)
        return stdout_buffer.getvalue(), error_output

    return stdout_buffer.getvalue(), stderr_buffer.getvalue()


def is_embedding_model(model: dict) -> bool:
    name = str(model.get("name", "")).lower()
    details = model.get("details") or {}
    family = str(details.get("family", "")).lower()
    families = [str(item).lower() for item in details.get("families", [])]

    embedding_markers = (
        "embed",
        "embedding",
        "bge",
        "e5",
        "nomic-bert",
        "snowflake-arctic-embed",
    )

    return any(marker in name for marker in embedding_markers) or any(
        marker in family or marker in families for marker in embedding_markers
    )


def load_local_model_options(
    base_url: str,
    default_chat_model: str,
    default_embedding_model: str,
) -> tuple[list[str], list[str], str | None]:
    try:
        with urlopen(f"{base_url}/api/tags") as response:
            payload = json.load(response)
    except URLError as exc:
        return [default_chat_model], [default_embedding_model], (
            f"Unable to load local Ollama models: {exc.reason}"
        )
    except Exception as exc:
        return [default_chat_model], [default_embedding_model], (
            f"Unable to load local Ollama models: {exc}"
        )

    models = payload.get("models", [])

    chat_model_names = sorted(
        {
            model["name"]
            for model in models
            if model.get("name") and not is_embedding_model(model)
        }
    )

    embedding_model_names = sorted(
        {
            model["name"]
            for model in models
            if model.get("name") and is_embedding_model(model)
        }
    )

    if not chat_model_names:
        chat_model_names = [default_chat_model]

    if not embedding_model_names:
        embedding_model_names = [default_embedding_model]

    if not models:
        return chat_model_names, embedding_model_names, "No local Ollama models were found."

    return chat_model_names, embedding_model_names, None


def get_embeddings(
    model_name: str,
    base_url: str,
    provider: str = "ollama",
    github_token: str | None = None,
) -> object:
    """
    Return embeddings instance based on provider selection.

    .. deprecated::
       Use ``fin_ai.core.embeddings.create_embeddings`` instead.
    """
    from fin_ai.core.embeddings import create_embeddings as _ce
    return _ce(
        provider=provider,
        model=model_name,
        api_base=base_url,
        api_key=github_token,
    )

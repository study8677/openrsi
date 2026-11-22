"""Thin, zero-extra-API-cost client wrapping local `codex exec` (ChatGPT OAuth)."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base."""
    return len(_ENC.encode(text))


@dataclass
class CodexRunResult:
    returncode: int
    last_message: str
    stdout: str
    stderr: str


def run_codex_in_workspace(
    prompt: str,
    workdir: Path,
    model: str | None = None,
    sandbox: str = "workspace-write",
    reasoning_effort: str = "low",
    timeout_sec: int = 240,
) -> CodexRunResult:
    """Execute `codex exec` non-interactively in an isolated workspace directory."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as out_f:
        out_path = Path(out_f.name)

    cmd = [
        "codex",
        "exec",
        "-C",
        str(workdir),
        "-s",
        sandbox,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "-o",
        str(out_path),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append("-")  # Read prompt from stdin

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        last_msg = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""
        return CodexRunResult(
            returncode=proc.returncode,
            last_message=last_msg,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CodexRunResult(
            returncode=-1,
            last_message="",
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"codex exec timed out after {timeout_sec}s",
        )
    finally:
        out_path.unlink(missing_ok=True)


def call_codex_text(
    prompt: str,
    model: str | None = None,
    timeout_sec: int = 180,
) -> str:
    """Run a read-only `codex exec` call in an empty temp dir and return its final text."""
    with tempfile.TemporaryDirectory(prefix="openrsi_codex_ro_") as tmpdir:
        res = run_codex_in_workspace(
            prompt=prompt,
            workdir=Path(tmpdir),
            model=model,
            sandbox="read-only",
            timeout_sec=timeout_sec,
        )
        if res.returncode != 0 and not res.last_message:
            raise RuntimeError(
                f"codex exec failed (rc={res.returncode}): {res.stderr.strip()[:500]}"
            )
        return res.last_message

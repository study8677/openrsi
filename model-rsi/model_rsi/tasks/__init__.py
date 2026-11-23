"""Verifiable task suites and immutable deterministic verifiers."""

from model_rsi.protocol import VerifiableTask, VerificationResult
from model_rsi.tasks.code_exec import verify_code_exec
from model_rsi.tasks.countdown import verify_countdown
from model_rsi.tasks.regression import verify_regression


def verify_response(task: VerifiableTask, response: str) -> VerificationResult:
    """Dispatch to the immutable deterministic verifier for a task's domain."""
    if task.domain == "countdown":
        return verify_countdown(task, response)
    if task.domain == "code_exec":
        return verify_code_exec(task, response)
    if task.domain == "regression":
        return verify_regression(task, response)
    raise ValueError(f"Unsupported task domain: {task.domain}")


__all__ = [
    "verify_response",
    "verify_countdown",
    "verify_code_exec",
    "verify_regression",
]

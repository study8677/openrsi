#!/bin/bash
set -u

mkdir -p /logs/verifier

# Run pytest and capture CTRF report for continuous partial-credit scoring (0.0 ~ 1.0)
pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
PYTEST_EXIT=$?

python3 - <<'PY'
import json
from pathlib import Path

ctrf_path = Path("/logs/verifier/ctrf.json")
reward_path = Path("/logs/verifier/reward.txt")

score = 0.0
if ctrf_path.exists():
    data = json.loads(ctrf_path.read_text(encoding="utf-8"))
    summary = data.get("results", {}).get("summary", {})
    total = summary.get("tests", 0)
    passed = summary.get("passed", 0)
    if total > 0:
        score = passed / total

reward_path.write_text(f"{score:.4f}\n", encoding="utf-8")
print(f"[verifier] passed={score:.4f}")
PY

exit 0

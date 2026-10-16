import json
import math
import re
from collections import Counter
from pathlib import Path


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def token_f1(left: str, right: str) -> float:
    a = Counter(normalize_answer(left).split())
    b = Counter(normalize_answer(right).split())
    overlap = sum((a & b).values())
    if not a or not b or overlap == 0:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2.0 * precision * recall / (precision + recall)


def load_rows(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"sample_id", "dataset", "model", "answer", "confidence", "correct"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        if row["correct"] not in (0, 1):
            raise ValueError(f"row {index}: correct must be 0 or 1")
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"row {index}: confidence must be finite and between 0 and 1")
        peers = row.get("peer_answers", [])
        weights = row.get("peer_confidences", [])
        if not isinstance(peers, list) or not isinstance(weights, list):
            raise ValueError(f"row {index}: peer answers and confidences must be lists")
        if weights and len(weights) != len(peers):
            raise ValueError(f"row {index}: peer answers and confidences must have equal lengths")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in weights):
            raise ValueError(f"row {index}: peer confidences must be finite and between 0 and 1")
    if not rows:
        raise ValueError("input contains no rows")
    return rows


def save_result(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

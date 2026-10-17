from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


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


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def aurc(labels: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    errors = 0
    risks = []
    for count, index in enumerate(order, start=1):
        errors += labels[index]
        risks.append(errors / count)
    return float(np.mean(risks))


def selective_accuracy(labels: list[int], scores: list[float], coverage: float) -> float:
    count = max(1, min(len(labels), round(len(labels) * coverage)))
    selected = sorted(range(len(scores)), key=lambda index: (scores[index], index))[:count]
    return 1.0 - sum(labels[index] for index in selected) / count


def metric_record(rows: list[dict], scores: list[float]) -> dict[str, float | int]:
    labels = [1 - int(row["correct"]) for row in rows]
    return {
        "samples": len(rows),
        "error_rate": float(np.mean(labels)),
        "aurc": aurc(labels, scores),
        "error_auroc": binary_auc(labels, scores),
        "accuracy_at_80_percent_coverage": selective_accuracy(labels, scores, 0.8),
    }


def normalized_entropy(answers: list[str]) -> float:
    """Entropy of normalized text counts, without semantic clustering."""
    if len(answers) < 2:
        return 0.0
    counts = Counter(normalize_answer(answer) for answer in answers)
    probabilities = [count / len(answers) for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(answers))


def features(row: dict, peer_limit: int | None = None) -> dict[str, float]:
    answer = str(row["answer"])
    peers = [str(value) for value in row.get("peer_answers", [])]
    peer_confidences = [float(value) for value in row.get("peer_confidences", [])]
    if peer_limit is not None:
        peers = peers[:peer_limit]
        peer_confidences = peer_confidences[:peer_limit]
    confidence = min(1.0, max(0.0, float(row["confidence"])))
    similarities = [token_f1(answer, peer) for peer in peers]
    disagreement = 1.0 - float(np.mean(similarities)) if similarities else 0.0
    if similarities and len(peer_confidences) == len(similarities):
        weights = np.asarray(peer_confidences, dtype=float)
        weights = np.maximum(weights, 1e-6)
        weighted = float(np.average(1.0 - np.asarray(similarities), weights=weights))
    else:
        weighted = disagreement
    ambiguity = normalized_entropy(peers)
    all_answers = [answer, *peers]
    majority = max(Counter(normalize_answer(value) for value in all_answers).values())
    variation = 1.0 - majority / len(all_answers)
    absolute = 1.0 - confidence
    # Fixed weights from the trained risk scorer.
    spar = 0.40 * absolute + 0.35 * weighted + 0.25 * ambiguity
    return {
        "absolute": absolute,
        "peer_disagreement": disagreement,
        "weighted_disagreement": weighted,
        "semantic_entropy": normalized_entropy(all_answers),
        "variation_ratio": variation,
        "ambiguity": ambiguity,
        "spar": spar,
    }


def risk_scores(rows: list[dict], method: str, peer_limit: int | None = None) -> list[float]:
    return [features(row, peer_limit)[method] for row in rows]


def evaluate(rows: list[dict], methods) -> dict[str, dict]:
    return {method: metric_record(rows, risk_scores(rows, method)) for method in methods}


def grouped(rows: list[dict], field: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return dict(sorted(groups.items()))

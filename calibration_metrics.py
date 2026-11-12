from __future__ import annotations

import numpy as np


def brier_score(labels: list[int], risks: list[float]) -> float:
    return float(np.mean([(label - risk) ** 2 for label, risk in zip(labels, risks)]))


def expected_calibration_error(labels: list[int], risks: list[float], bins: int = 10) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = [
            position for position, risk in enumerate(risks)
            if low <= risk < high or (index == bins - 1 and risk == 1.0)
        ]
        if selected:
            observed = float(np.mean([labels[position] for position in selected]))
            predicted = float(np.mean([risks[position] for position in selected]))
            error += len(selected) / total * abs(observed - predicted)
    return error

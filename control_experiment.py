import argparse

import numpy as np

from common import features, load_rows, metric_record, risk_scores, save_result, selective_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="outputs/controls.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    labels = [1 - int(row["correct"]) for row in rows]
    spar = risk_scores(rows, "spar")
    capacity = {}
    maximum = max((len(row.get("peer_answers", [])) for row in rows), default=0)
    for count in range(1, maximum + 1):
        scores = [features(row, count)["spar"] for row in rows]
        capacity[str(count)] = metric_record(rows, scores)
    correct_scores = [score for score, label in zip(spar, labels) if label == 0]
    error_scores = [score for score, label in zip(spar, labels) if label == 1]
    deployment = {str(value): selective_accuracy(labels, spar, value) for value in (0.5, 0.7, 0.8, 0.9, 1.0)}
    result = {
        "demo": args.input is None,
        "peer_capacity": capacity,
        "correctness": {
            "mean_risk_correct": float(np.mean(correct_scores)) if correct_scores else None,
            "mean_risk_error": float(np.mean(error_scores)) if error_scores else None,
            "all_rows": metric_record(rows, spar),
        },
        "deployment_accuracy_by_coverage": deployment,
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()


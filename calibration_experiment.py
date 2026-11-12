import argparse

from calibration_metrics import brier_score, expected_calibration_error
from common import grouped, load_rows, risk_scores, save_result


METHODS = ["absolute", "weighted_disagreement", "spar"]


def evaluate(rows: list[dict]) -> dict:
    labels = [1 - int(row["correct"]) for row in rows]
    result = {}
    for method in METHODS:
        risks = risk_scores(rows, method)
        result[method] = {
            "brier": brier_score(labels, risks),
            "ece": expected_calibration_error(labels, risks),
            "mean_predicted_error": sum(risks) / len(risks),
            "observed_error": sum(labels) / len(labels),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="outputs/calibration.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    result = {
        "demo": args.input is None,
        "overall": evaluate(rows),
        "by_dataset": {name: evaluate(group) for name, group in grouped(rows, "dataset").items()},
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()


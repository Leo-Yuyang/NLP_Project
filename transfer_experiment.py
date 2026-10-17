import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import features, load_rows, metric_record, risk_scores, save_result


FEATURE_NAMES = ["absolute", "weighted_disagreement", "ambiguity"]


def matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[features(row)[name] for name in FEATURE_NAMES] for row in rows], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Saved answers in JSONL format; omit to use synthetic demo data")
    parser.add_argument("--source", default="vqav2,aokvqa")
    parser.add_argument("--target", default="textvqa,vizwiz")
    parser.add_argument("--output", default="outputs/transfer.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    source_names = {name.strip() for name in args.source.split(",") if name.strip()}
    target_names = {name.strip() for name in args.target.split(",") if name.strip()}
    if source_names & target_names:
        parser.error("source and target datasets must not overlap")
    source = [row for row in rows if row["dataset"] in source_names]
    target = [row for row in rows if row["dataset"] in target_names]
    if not source or not target:
        parser.error("source and target datasets must both contain rows")
    labels = np.asarray([1 - int(row["correct"]) for row in source])
    if len(np.unique(labels)) < 2:
        parser.error("source data must contain both correct and incorrect answers")
    model = LogisticRegression(penalty="l2", random_state=42, max_iter=500).fit(matrix(source), labels)
    learned = model.predict_proba(matrix(target))[:, 1].tolist()
    result = {
        "demo": args.input is None,
        "source_datasets": sorted(source_names),
        "target_datasets": sorted(target_names),
        "features": FEATURE_NAMES,
        "penalty": "l2",
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "absolute": metric_record(target, risk_scores(target, "absolute")),
        "fixed_spar": metric_record(target, risk_scores(target, "spar")),
        "learned_transfer": metric_record(target, learned),
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()


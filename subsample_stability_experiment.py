import argparse

import numpy as np

from common import demo_rows, load_rows, metric_record, risk_scores, save_result


def evaluate(rows: list[dict]) -> float:
    baseline = metric_record(rows, risk_scores(rows, "absolute"))["aurc"]
    method = metric_record(rows, risk_scores(rows, "spar"))["aurc"]
    return (baseline - method) / baseline if baseline else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Random seeds for synthetic generation or subsampling")
    parser.add_argument("--output", default="outputs/subsample_stability.json")
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if not seeds:
        parser.error("--seeds must not be empty")
    if args.input is None:
        values = [evaluate(demo_rows(seed)) for seed in seeds]
    else:
        rows = load_rows(args.input)
        values = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sample_size = min(len(rows), max(2, round(0.8 * len(rows))))
            indices = rng.choice(len(rows), size=sample_size, replace=False)
            values.append(evaluate([rows[int(index)] for index in indices]))
    result = {
        "demo": args.input is None,
        "scope": "synthetic_generation" if args.input is None else "row_subsampling",
        "seeds": seeds,
        "relative_aurc_reductions": values,
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

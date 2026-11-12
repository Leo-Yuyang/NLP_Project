import argparse
import random

from common import aurc, load_rows, risk_scores, save_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/significance.json")
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    rows = load_rows(args.input)
    labels = [1 - int(row["correct"]) for row in rows]
    absolute = risk_scores(rows, "absolute")
    spar = risk_scores(rows, "spar")
    rng = random.Random(args.seed)
    differences = []
    for _ in range(args.resamples):
        indices = [rng.randrange(len(rows)) for _ in rows]
        sampled_labels = [labels[index] for index in indices]
        baseline = aurc(sampled_labels, [absolute[index] for index in indices])
        method = aurc(sampled_labels, [spar[index] for index in indices])
        differences.append(baseline - method)
    differences.sort()
    low = differences[int(0.025 * len(differences))]
    high = differences[min(len(differences) - 1, int(0.975 * len(differences)))]
    result = {
        "demo": args.input is None,
        "resamples": args.resamples,
        "mean_aurc_improvement": sum(differences) / len(differences),
        "confidence_interval_95": [low, high],
        "probability_of_improvement": sum(value > 0 for value in differences) / len(differences),
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()


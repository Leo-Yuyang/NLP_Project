import argparse

from common import evaluate, grouped, load_rows, save_result


METHODS = [
    "absolute",
    "variation_ratio",
    "semantic_entropy",
    "peer_disagreement",
    "weighted_disagreement",
    "spar",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Saved answers in JSONL format; omit to use synthetic demo data")
    parser.add_argument("--output", default="outputs/baselines.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    result = {
        "demo": args.input is None,
        "overall": evaluate(rows, METHODS),
        "by_dataset": {name: evaluate(group, METHODS) for name, group in grouped(rows, "dataset").items()},
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()


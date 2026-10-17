import argparse

from common import evaluate, grouped, load_rows, save_result


METHODS = ["absolute", "peer_disagreement", "semantic_entropy", "spar"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/main.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    result = {
        "demo": False,
        "overall": evaluate(rows, METHODS),
        "by_dataset": {name: evaluate(group, METHODS) for name, group in grouped(rows, "dataset").items()},
        "by_model": {name: evaluate(group, METHODS) for name, group in grouped(rows, "model").items()},
    }
    save_result(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()


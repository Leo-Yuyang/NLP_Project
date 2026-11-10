import argparse

from common import evaluate, grouped, load_rows, save_result


METHODS = ["absolute", "spar"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="outputs/scale.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    if any("model_size_b" not in row for row in rows):
        raise ValueError("scale experiment requires model_size_b")
    cells = []
    for size, group in sorted(grouped(rows, "model_size_b").items(), key=lambda item: float(item[0])):
        metrics = evaluate(group, METHODS)
        baseline = metrics["absolute"]["aurc"]
        gain = (baseline - metrics["spar"]["aurc"]) / baseline if baseline else 0.0
        cells.append({"model_size_b": float(size), "relative_aurc_reduction": gain, "metrics": metrics})
    save_result(args.output, {"demo": args.input is None, "scale_results": cells})
    print(args.output)


if __name__ == "__main__":
    main()


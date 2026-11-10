import argparse

from common import features, load_rows, metric_record, save_result


def scores(rows: list[dict]) -> dict[str, list[float]]:
    values = [features(row) for row in rows]
    return {
        "absolute_only": [value["absolute"] for value in values],
        "peer_only": [value["weighted_disagreement"] for value in values],
        "without_ambiguity": [
            (0.40 * value["absolute"] + 0.35 * value["weighted_disagreement"]) / 0.75
            for value in values
        ],
        "unweighted_peers": [0.40 * value["absolute"] + 0.35 * value["peer_disagreement"] + 0.25 * value["ambiguity"] for value in values],
        "full_spar": [value["spar"] for value in values],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="outputs/ablation.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    result = {name: metric_record(rows, values) for name, values in scores(rows).items()}
    save_result(args.output, {"demo": args.input is None, "ablations": result})
    print(args.output)


if __name__ == "__main__":
    main()


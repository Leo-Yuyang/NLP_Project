import argparse
import copy
import random

from common import load_rows, metric_record, risk_scores, save_result


def confidence_noise(rows: list[dict], standard_deviation: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["confidence"] = min(1.0, max(0.0, float(row["confidence"]) + rng.gauss(0.0, standard_deviation)))
    return changed


def peer_corruption(rows: list[dict], count: int) -> list[dict]:
    changed = copy.deepcopy(rows)
    for row in changed:
        peers = row.get("peer_answers", [])
        for index in range(min(count, len(peers))):
            peers[index] = f"corrupted_{index}"
            if index < len(row.get("peer_confidences", [])):
                row["peer_confidences"][index] = 0.5
    return changed


def evaluate(rows: list[dict]) -> dict:
    return metric_record(rows, risk_scores(rows, "spar"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="outputs/robustness.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    confidence = {str(level): evaluate(confidence_noise(rows, level, 42)) for level in (0.0, 0.05, 0.10, 0.20)}
    maximum = max((len(row.get("peer_answers", [])) for row in rows), default=0)
    corruption = {str(count): evaluate(peer_corruption(rows, count)) for count in range(maximum + 1)}
    save_result(args.output, {"demo": args.input is None, "confidence_noise": confidence, "peer_corruption": corruption})
    print(args.output)


if __name__ == "__main__":
    main()


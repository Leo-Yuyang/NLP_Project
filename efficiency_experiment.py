import argparse
import time

from common import load_rows, risk_scores, save_result


METHODS = ["absolute", "peer_disagreement", "semantic_entropy", "spar"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure risk scoring time on saved answers, excluding VLM inference.")
    parser.add_argument("--input", help="Saved answers in JSONL format; omit to use synthetic demo data")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="outputs/efficiency.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows = load_rows(args.input)
    timings = {}
    for method in METHODS:
        risk_scores(rows, method)
        started = time.perf_counter()
        for _ in range(args.repeats):
            risk_scores(rows, method)
        elapsed = time.perf_counter() - started
        timings[method] = {
            "seconds": elapsed,
            "rows_per_second": len(rows) * args.repeats / elapsed,
        }
    save_result(args.output, {"demo": args.input is None, "scope": "scoring_only", "rows": len(rows), "repeats": args.repeats, "timings": timings})
    print(args.output)


if __name__ == "__main__":
    main()


import re
from collections import Counter


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def token_f1(left: str, right: str) -> float:
    a = Counter(normalize_answer(left).split())
    b = Counter(normalize_answer(right).split())
    overlap = sum((a & b).values())
    if not a or not b or overlap == 0:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2.0 * precision * recall / (precision + recall)

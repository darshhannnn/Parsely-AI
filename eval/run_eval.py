"""
Retrieval evaluation harness for Parsely AI.

Measures retrieval hit rate against eval/eval_set.json: for each question the
retriever must select, from the source document, a context that contains the
labelled evidence string. Hit rate is the fraction of questions where it does.

Usage:
    python eval/run_eval.py                                  # all retrievers
    python eval/run_eval.py --retrievers baseline semantic   # subset
    python eval/run_eval.py --budget 4000 --out eval/RESULTS.md

This is the harness behind the before/after numbers reported in the README
("Improved retrieval hit rate from X% to Y% ...").
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import build_retriever, normalize_ws  # noqa: E402

RETRIEVER_ORDER = ["baseline", "semantic", "reranked"]
RETRIEVER_LABELS = {
    "baseline": "Fixed char-window chunks + TF-IDF (baseline)",
    "semantic": "Semantic clause-aware chunking + TF-IDF",
    "reranked": "Semantic chunking + embedding re-ranker (RRF)",
}


def load_eval_set(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    documents = {}
    invalid = []
    for item in items:
        doc = item["document"]
        if doc not in documents:
            doc_path = EVAL_DIR / "documents" / f"{doc}.txt"
            if not doc_path.exists():
                raise FileNotFoundError(f"Document text missing: {doc_path}")
            documents[doc] = normalize_ws(doc_path.read_text(encoding="utf-8"))
        if normalize_ws(item["evidence"]) not in documents[doc]:
            invalid.append(item["id"])
    if invalid:
        raise ValueError(
            f"Evidence strings not found in their documents (bad eval data): {invalid}"
        )
    return items, documents


def run_retriever(name: str, items, documents, budget_chars: int):
    retriever = build_retriever(name)
    hits = []
    per_category = defaultdict(lambda: [0, 0])
    start = time.time()
    for item in items:
        context = retriever.retrieve(
            documents[item["document"]], item["question"], budget_chars
        )
        hit = normalize_ws(item["evidence"]) in normalize_ws(context)
        hits.append((item, hit))
        per_category[item["category"]][0] += int(hit)
        per_category[item["category"]][1] += 1
    duration = time.time() - start
    return hits, per_category, duration


def main():
    parser = argparse.ArgumentParser(description="Run the retrieval evaluation harness")
    parser.add_argument(
        "--retrievers", nargs="+", default=RETRIEVER_ORDER, choices=RETRIEVER_ORDER
    )
    parser.add_argument("--budget", type=int, default=6000, help="Context budget in characters")
    parser.add_argument("--out", type=Path, default=None, help="Write a markdown report here")
    args = parser.parse_args()

    items, documents = load_eval_set(EVAL_DIR / "eval_set.json")
    print(f"Eval set: {len(items)} questions across {len(documents)} documents")
    print(f"Context budget per question: {args.budget} chars\n")

    results = {}
    for name in args.retrievers:
        print(f"Running retriever: {name} ...")
        hits, per_category, duration = run_retriever(name, items, documents, args.budget)
        hit_count = sum(1 for _, h in hits if h)
        results[name] = {"hits": hits, "per_category": per_category,
                         "hit_count": hit_count, "total": len(items), "duration": duration}
        print(f"  -> {hit_count}/{len(items)} hits ({100.0 * hit_count / len(items):.1f}%) "
              f"in {duration:.1f}s\n")

    # Console table
    print("=" * 72)
    print(f"{'Retriever':<46} {'Hit rate':>10} {'Time':>8}")
    print("-" * 72)
    for name in args.retrievers:
        r = results[name]
        label = RETRIEVER_LABELS[name]
        print(f"{label[:45]:<46} {r['hit_count']}/{r['total']:>4} "
              f"({100.0 * r['hit_count'] / r['total']:.1f}%) {r['duration']:>6.1f}s")
    print("=" * 72)

    # Per-category breakdown for the best retriever
    best = max(results, key=lambda n: results[n]["hit_count"])
    print(f"\nPer-category breakdown for best retriever ({best}):")
    for category, (got, total) in sorted(results[best]["per_category"].items()):
        print(f"  {category:<18} {got:>3}/{total:<3}")

    # Misses for the best retriever, to guide further work
    misses = [item["id"] for item, hit in results[best]["hits"] if not hit]
    if misses:
        print(f"\nRemaining misses: {', '.join(misses)}")

    if args.out:
        lines = [
            "# Retrieval Evaluation Results",
            "",
            f"- Eval set: {len(items)} questions across {len(documents)} insurance policy documents",
            f"- Context budget per question: {args.budget} characters",
            "- Metric: **hit rate** — the labelled evidence clause is contained in the retrieved context",
            "",
            "| Retriever | Hit rate |",
            "|---|---|",
        ]
        for name in args.retrievers:
            r = results[name]
            lines.append(
                f"| {RETRIEVER_LABELS[name]} | {r['hit_count']}/{r['total']} "
                f"(**{100.0 * r['hit_count'] / r['total']:.1f}%**) |"
            )
        lines += ["", f"Best per-category breakdown ({best}):", ""]
        for category, (got, total) in sorted(results[best]["per_category"].items()):
            lines.append(f"- {category}: {got}/{total}")
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMarkdown report written to {args.out}")


if __name__ == "__main__":
    main()

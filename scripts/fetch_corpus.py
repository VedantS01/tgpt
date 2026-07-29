"""Entrypoint — build the model corpus (Milestone 2).

    python -m scripts.fetch_corpus                      # the full mixture, ~2.4 GB
    python -m scripts.fetch_corpus --total-mb 400       # a small one, same proportions
    python -m scripts.fetch_corpus --only cpp wikipedia # just these, at full budget
    python -m scripts.fetch_corpus --list               # show the mixture and exit

Writes one file per source to `data/corpus/`, plus a ~1 MB held-out slice of each
to `data/heldout/` that nothing ever trains on. Re-running is cheap: a source
already at its budget is skipped, so an interrupted fetch resumes where it left off
(use `--force` to refetch anyway).

The source list, the byte budgets and the reasoning behind both live in
`tgpt/corpus.py`.
"""

import argparse
import os

from tgpt.corpus import OUT_DIR, SOURCES, fetch_all, report


def show_list(total_mb: float | None):
    nominal = sum(s.budget_mb for s in SOURCES)
    scale = (total_mb / nominal) if total_mb else 1.0
    print(f"{'source':<16}{'group':<11}{'MB':>7}   dataset")
    for group in ("code", "wikipedia", "help", "qa"):
        for s in SOURCES:
            if s.group == group:
                print(f"{s.name:<16}{s.group:<11}{s.budget_mb*scale:7.0f}   "
                      f"{s.dataset or 'github release tarballs'}")
    by_group = {}
    for s in SOURCES:
        by_group[s.group] = by_group.get(s.group, 0) + s.budget_mb * scale
    total = sum(by_group.values())
    print(f"\n{'total':<16}{'':<11}{total:7.0f} MB")
    for g, mb in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"  {g:<12} {mb:7.0f} MB  {mb/total*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-mb", type=float, default=None,
                    help="rescale every source proportionally to hit this total")
    ap.add_argument("--only", nargs="+", default=None, help="fetch only these sources")
    ap.add_argument("--force", action="store_true", help="refetch sources already at budget")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print the mixture and exit")
    args = ap.parse_args()

    if args.list:
        show_list(args.total_mb)
        return

    written = fetch_all(args.total_mb, args.only, args.force, args.workers)
    report(written)
    print(f"\ncorpus -> {OUT_DIR}/    "
          f"({len([f for f in os.listdir(OUT_DIR) if f.endswith('.txt')])} files)")
    print("next: python -m scripts.train_code_tokenizer && python -m scripts.prepare_data")


if __name__ == "__main__":
    main()

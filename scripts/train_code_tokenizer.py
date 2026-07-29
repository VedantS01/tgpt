"""Entrypoint — train the code-aware byte-level BPE tokenizer (Milestone 1b).

    python -m scripts.train_code_tokenizer [--vocab-size 32000]

Assumes the corpus exists (`python -m scripts.fetch_corpus`).

The corpus is far larger than a tokenizer needs, so it is subsampled
proportionally: merge rankings for a 32k vocabulary settle long before 300 MB,
and reading all of it would cost hours for a vocabulary that does not change.
"""

import argparse
import glob
import os

from tgpt import langspec
from tgpt.code_tokenizer import ensure_single_tokens, train_code_tokenizer

CORPUS_GLOB = "data/corpus/*.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-split-digits", action="store_true")
    ap.add_argument("--sample-mb", type=float, default=300,
                    help="total bytes to train on, sampled proportionally (0 = all)")
    ap.add_argument("--corpus", default=CORPUS_GLOB)
    args = ap.parse_args()

    out = args.out or f"tokenizer/tgpt-code-{args.vocab_size}.json"
    files = sorted(glob.glob(args.corpus))
    if not files:
        raise SystemExit(f"no corpus matched {args.corpus} — run: python -m scripts.fetch_corpus")

    total = sum(os.path.getsize(f) for f in files) / 1e6
    print(f"training vocab={args.vocab_size} on {len(files)} files ({total:.0f} MB): "
          f"{[os.path.basename(f)[:-4] for f in files]}")
    train_code_tokenizer(
        files, out, vocab_size=args.vocab_size, split_digits=not args.no_split_digits,
        sample_bytes=args.sample_mb * 1e6 or None,
    )

    # Requirement 1 is a hard promise, so back it with an explicit guarantee for
    # any rung the corpus did not already teach. Runs of >= 3 spaces are safe to
    # force: they essentially never occur in prose, so nothing else regresses.
    ladder = [" " * n for n in langspec.SPACE_LADDER if n >= 3]
    added = ensure_single_tokens(out, ladder)
    print(f"space ladder force-added: {[len(a) for a in added] or 'none (all learned)'}")
    print(f"tokenizer -> {out}")


if __name__ == "__main__":
    main()

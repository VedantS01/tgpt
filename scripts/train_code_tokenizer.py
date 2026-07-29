"""Entrypoint — train the code-aware byte-level BPE tokenizer (Milestone 1b).

    python -m scripts.train_code_tokenizer [--vocab-size 32000]

Assumes the corpus exists (`python -m scripts.fetch_code_corpus`).
"""

import argparse
import os

from tgpt import langspec
from tgpt.code_tokenizer import ensure_single_tokens, train_code_tokenizer

CORPUS_DIR = "data/code"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-split-digits", action="store_true")
    args = ap.parse_args()

    out = args.out or f"tokenizer/tgpt-code-{args.vocab_size}.json"
    files = sorted(
        os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")
    )
    if not files:
        raise SystemExit(f"no corpus in {CORPUS_DIR}/ — run: python -m scripts.fetch_code_corpus")

    print(f"training vocab={args.vocab_size} on {len(files)} files: {[os.path.basename(f) for f in files]}")
    train_code_tokenizer(
        files, out, vocab_size=args.vocab_size, split_digits=not args.no_split_digits
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

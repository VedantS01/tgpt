"""Entrypoint — tokenize the corpus into binary shards (Milestone 2).

    python -m scripts.prepare_data                       # code+prose mixture, code tokenizer
    python -m scripts.prepare_data --tokenizer tokenizer/tgpt.model --out data/wiki
    python -m scripts.prepare_data --sources data/corpus.txt

Produces `<out>/train.bin`, `<out>/val.bin` (uint16 token ids) and `<out>/meta.json`.

The default mixture is `data/code/*.txt` — Python, C++, JavaScript and Wikipedia
prose — which is the corpus the M1b tokenizer was built for. Because the shards
are keyed to a tokenizer, swapping tokenizers means re-running this with a
different `--out`; keeping both around is what makes the M1-vs-M1b comparison
measurable at the loss level rather than only in bytes per token.
"""

import argparse
import glob
import os

from tgpt.data import prepare

DEFAULT_SOURCES = "data/corpus/*.txt"
CODE_TOKENIZER = "tokenizer/tgpt-code-32000.json"
WIKI_TOKENIZER = "tokenizer/tgpt.model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=[DEFAULT_SOURCES],
                    help="corpus files or globs (default: data/corpus/*.txt)")
    ap.add_argument("--tokenizer", default=None,
                    help=f"default: {CODE_TOKENIZER}, falling back to {WIKI_TOKENIZER}")
    ap.add_argument("--out", default="data/shards", help="output directory for the .bin shards")
    ap.add_argument("--val-fraction", type=float, default=0.01,
                    help="share of corpus BYTES held out, clamped to 4-16 MB")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tokenizer = args.tokenizer
    if tokenizer is None:
        tokenizer = CODE_TOKENIZER if os.path.exists(CODE_TOKENIZER) else WIKI_TOKENIZER
    if not os.path.exists(tokenizer):
        raise SystemExit(f"{tokenizer} not found — train a tokenizer first (M1 / M1b).")

    paths = sorted({p for pattern in args.sources for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(
            f"no corpus files matched {args.sources} — run: python -m scripts.fetch_corpus"
        )

    prepare(paths, tokenizer, args.out, val_fraction=args.val_fraction, seed=args.seed)
    print(f"\ntoken shards -> {args.out}/train.bin, {args.out}/val.bin, {args.out}/meta.json")


if __name__ == "__main__":
    main()

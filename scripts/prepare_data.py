"""Entrypoint — tokenize the corpus into binary shards (Milestone 2).

    python -m scripts.prepare_data

Requires the tokenizer from Milestone 1. Produces data/train.bin and data/val.bin.
"""

import os
from tgpt.data import download_corpus, prepare

CORPUS_PATH = "data/corpus.txt"
TOKENIZER_PATH = "tokenizer/tgpt.model"
OUT_DIR = "data"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CORPUS_PATH):
        download_corpus(CORPUS_PATH, source="wikitext-103")
    prepare(CORPUS_PATH, TOKENIZER_PATH, OUT_DIR)
    print(f"token shards written -> {OUT_DIR}/train.bin, {OUT_DIR}/val.bin")


if __name__ == "__main__":
    main()

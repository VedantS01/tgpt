"""Entrypoint — train the SentencePiece tokenizer (Milestone 1).

    python -m scripts.train_tokenizer

Thin wrapper: it ensures the corpus exists, then calls into tgpt.tokenizer. Implement the
TODO blocks in tgpt/data.py (download) and tgpt/tokenizer.py (train) and this runs as-is.
"""

import os
from tgpt.data import download_corpus
from tgpt.tokenizer import train_tokenizer

CORPUS_PATH = "data/corpus.txt"
MODEL_PREFIX = "tokenizer/tgpt"
VOCAB_SIZE = 16000


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("tokenizer", exist_ok=True)
    if not os.path.exists(CORPUS_PATH):
        download_corpus(CORPUS_PATH, source="wikitext-103")
    model_path = train_tokenizer(CORPUS_PATH, MODEL_PREFIX, vocab_size=VOCAB_SIZE)
    print(f"tokenizer trained -> {model_path}")


if __name__ == "__main__":
    main()

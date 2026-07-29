"""Milestone 2 — Data pipeline: corpus -> token shards -> batches.

The professional pattern: tokenize the whole corpus ONCE into a flat array of integer ids,
store it on disk as a compact binary file, and at train time read random windows from it via
a memory-map. memmap lets you train on a corpus far larger than RAM — the OS pages in only the
slices you touch. Token ids fit in uint16 when vocab_size < 65,536 (ours is 16k), which halves
disk + I/O versus int32.

Three pieces to implement:
    download_corpus(...)  -> a plain-text file on disk (Wikipedia / WikiText)
    prepare(...)          -> read text, tokenize, write train.bin / val.bin (uint16)
    get_batch(...)        -> sample a random (x, y) batch from a .bin memmap
"""

from __future__ import annotations
import os
import numpy as np
import torch

from .tokenizer import Tokenizer


def download_corpus(out_path: str, source: str = "wikitext-103") -> str:
    """Download a text corpus to `out_path` and return it.

    Start small for the first end-to-end run, then scale the corpus without changing
    anything downstream.

    WikiText-103 ships as one line per paragraph, with article headings as their own
    lines (" = Title = "). We write the lines through as-is: the text stays one stream,
    headings mark where articles begin, and the tokenize step (M2) inserts an EOS token
    between documents so the model can learn boundaries.

    Returns out_path.
    """
    from datasets import load_dataset

    if source == "wikitext-103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    elif source == "wikipedia":
        # The full dump is ~20GB of text — slice it for the laptop phase.
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:2%]")
    else:
        raise ValueError(f"unknown corpus source: {source}")

    tmp_path = out_path + ".tmp"
    n_chars = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"]
            if not text.strip():
                continue
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            n_chars += len(text)
    os.replace(tmp_path, out_path)  # atomic: no half-written corpus if interrupted
    print(f"corpus written -> {out_path} ({n_chars/1e6:.0f}M chars)")
    return out_path


def prepare(corpus_path: str, tokenizer_path: str, out_dir: str, val_fraction: float = 0.0005) -> None:
    """Tokenize `corpus_path` and write `train.bin` and `val.bin` (uint16) into `out_dir`.

    TODO(M2):
      - tok = Tokenizer(tokenizer_path)
      - read the corpus (stream it if large — do not hold the whole tokenized array in RAM
        if you can avoid it; you can write in chunks with np.memmap or append to a file).
      - encode to ids (uint16). Insert your document separator / eos between documents.
      - split off the last `val_fraction` as validation.
      - save with ids.astype(np.uint16).tofile(path)  (read back with np.memmap(..., uint16)).
      - print the token counts so you can sanity-check (Chinchilla: ~20 tokens/param is "enough").
    """
    raise NotImplementedError("M2: tokenize + shard the corpus")


def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    """Return one (x, y) batch of token windows from `<split>.bin`.

    x = tokens[i : i+block_size]      (the context)
    y = tokens[i+1 : i+block_size+1]  (each position's *next* token — the target)

    TODO(M2):
      - data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
      - ix = torch.randint(len(data) - block_size, (batch_size,))
      - stack the windows into x and y (cast to int64 for the embedding lookup).
      - move to device. For MPS/CUDA, pinning + non_blocking is a nice touch.
    """
    raise NotImplementedError("M2: sample a batch from the memmap")

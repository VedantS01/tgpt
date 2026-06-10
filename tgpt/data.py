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

    TODO(M2):
      - "wikitext-103": from datasets import load_dataset;
                        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
                        write the article text to out_path (one stream of text).
      - "wikipedia":    load_dataset("wikimedia/wikipedia", "20231101.en") for the full dump
                        (large — take a slice for the laptop phase).
      - Decide how to separate documents (e.g. a blank line, or a BOS/EOS token at tokenize time)
        so the model learns document boundaries.

    Returns out_path.
    """
    raise NotImplementedError("M2: download the corpus")


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

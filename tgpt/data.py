"""Milestone 2 — Data pipeline: corpus -> token shards -> batches.

The professional pattern: tokenize the whole corpus ONCE into a flat array of integer ids,
store it on disk as a compact binary file, and at train time read random windows from it via
a memory-map. memmap lets you train on a corpus far larger than RAM — the OS pages in only the
slices you touch. Token ids fit in uint16 when vocab_size < 65,536 (ours is 16k-32k), which
halves disk + I/O versus int32.

Three pieces:
    download_corpus(...)  -> a plain-text file on disk (Wikipedia / WikiText)
    prepare(...)          -> read text, tokenize, write train.bin / val.bin (uint16)
    get_batch(...)        -> sample a random (x, y) batch from a .bin memmap

Two decisions in here are worth more than the code around them:

**Documents are a real unit, so the corpus stores where they end.** Every corpus
file is a concatenation of documents separated by a NUL byte (`DOC_SEP`). NUL is
the one byte guaranteed not to occur in the text itself, and it never reaches the
tokenizer — `prepare` splits on it and replaces each boundary with the EOS token.
Without that, the model has no signal for "this text is finished", which is
exactly what generation needs in order to stop.

**Documents are shuffled before the train/val split, not after.** The corpus is a
mixture (Python, C++, JavaScript, prose), and each source is contiguous on disk.
Slicing the last 0.5% of the *token array* as validation would hand you a
validation set made entirely of whichever source happens to land last — you would
be measuring one language and calling it the loss. Shuffling whole documents
first makes both splits the same mixture, and keeping the shuffle at document
granularity means no document is half in train and half in val.
"""

from __future__ import annotations
import json
import os
import random

import numpy as np
import torch

from .tokenizer import load_tokenizer

# Document separator. NUL is deliberate: it is the only byte that never appears in
# real text, so splitting on it can never be ambiguous. It is a *corpus file*
# convention only — it is stripped before tokenization and never becomes a token.
DOC_SEP = "\x00"
_DOC_SEP_B = b"\x00"

# uint16 holds 0..65535. Anything larger needs uint32 and doubles the shard size.
_MAX_UINT16_VOCAB = 65536

# TrainConfig's defaults sample eval_iters * batch_size * block_size = 100*32*256
# tokens per evaluation. A validation set smaller than that is being resampled.
_MIN_VAL_TOKENS = 100 * 32 * 256


def download_corpus(out_path: str, source: str = "wikitext-103") -> str:
    """Download a text corpus to `out_path` and return it.

    Start small for the first end-to-end run, then scale the corpus without changing
    anything downstream.

    WikiText-103 ships as one line per paragraph, with article headings as their own
    lines (" = Title = "). Those headings are the only article boundary the dataset
    gives us, so we detect the top-level ones — exactly one `=` per side, since
    ` = = Section = = ` is a subsection of the same article — and emit a DOC_SEP
    before each. That turns a flat stream of paragraphs back into documents.

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
    n_chars = n_docs = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"]
            if not text.strip():
                continue
            if source == "wikipedia" or _is_article_heading(text):
                if n_docs:
                    f.write(DOC_SEP)
                n_docs += 1
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            n_chars += len(text)
    os.replace(tmp_path, out_path)  # atomic: no half-written corpus if interrupted
    print(f"corpus written -> {out_path} ({n_chars/1e6:.0f}M chars, {n_docs} documents)")
    return out_path


def _is_article_heading(line: str) -> bool:
    """True for WikiText's top-level ' = Title = ' lines, false for ' = = Section = = '."""
    s = line.strip()
    return s.startswith("= ") and s.endswith(" =") and not s.startswith("= =")


# --- corpus -> token shards --------------------------------------------------


def _index_documents(paths: list[str]) -> list[tuple[int, int, int]]:
    """Scan the corpus for document boundaries without loading it into memory.

    Returns (file_index, byte_offset, byte_length) per document. Only offsets are
    held in RAM — a 700 MB corpus indexes to a few MB of tuples — and the text
    itself is read back later, one batch of documents at a time.
    """
    docs: list[tuple[int, int, int]] = []
    for fi, path in enumerate(paths):
        size = os.path.getsize(path)
        start = pos = 0
        seen_sep = False
        with open(path, "rb") as f:
            while chunk := f.read(1 << 23):
                idx = chunk.find(_DOC_SEP_B)
                while idx != -1:
                    end = pos + idx
                    if end > start:
                        docs.append((fi, start, end - start))
                    start = end + 1
                    seen_sep = True
                    idx = chunk.find(_DOC_SEP_B, idx + 1)
                pos += len(chunk)
        if size > start:
            docs.append((fi, start, size - start))
        if not seen_sep:
            print(f"  note: {os.path.basename(path)} has no {DOC_SEP!r} separators "
                  f"— treating the whole file as one document")
    return docs


def _write_shard(
    handles: list, docs: list[tuple[int, int, int]], tok, out_path: str,
    per_file_tokens: list[int], batch_docs: int = 1024, flush_tokens: int = 1 << 23,
) -> int:
    """Tokenize `docs` in order and stream them into `out_path` as uint16. Returns token count."""
    from tqdm import tqdm

    n_tokens = 0
    buf: list[int] = []
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        for i in tqdm(range(0, len(docs), batch_docs), desc=os.path.basename(out_path), unit="batch"):
            batch = docs[i:i + batch_docs]
            texts = []
            for fi, start, length in batch:
                handles[fi].seek(start)
                # errors="replace" rather than "strict": a handful of mojibake bytes
                # in a 700 MB scrape should not abort a 20-minute tokenize.
                texts.append(handles[fi].read(length).decode("utf-8", errors="replace"))
            # add_eos, not add_bos: every document ends with the separator token, and
            # the *next* document's first token is then a natural "start" signal. One
            # marker is enough, and EOS is the one generation needs.
            for (fi, _, _), ids in zip(batch, tok.encode_batch(texts, add_eos=True)):
                per_file_tokens[fi] += len(ids)
                buf.extend(ids)
            if len(buf) >= flush_tokens:
                np.asarray(buf, dtype=np.uint16).tofile(out)
                n_tokens += len(buf)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(out)
            n_tokens += len(buf)
    os.replace(tmp, out_path)  # atomic, same as the corpus download
    return n_tokens


def prepare(
    corpus_path: str | list[str],
    tokenizer_path: str,
    out_dir: str,
    val_fraction: float = 0.04,
    seed: int = 1337,
) -> dict:
    """Tokenize the corpus and write `train.bin` + `val.bin` (uint16) into `out_dir`.

    `corpus_path` may be a single file or a list — the list form is the mixture case
    (`data/code/python.txt`, `.../cpp.txt`, `.../prose.txt`, ...). Documents from all
    sources are pooled and shuffled together, so train and val hold the same mix.

    Also writes `meta.json`, which records the tokenizer, its vocabulary size and the
    token counts. Training reads that instead of trusting a hardcoded `vocab_size`:
    a model built with the wrong vocabulary trains happily and produces garbage, and
    the check that prevents it costs one file.
    """
    paths = [corpus_path] if isinstance(corpus_path, str) else list(corpus_path)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"corpus file(s) not found: {missing}")

    tok = load_tokenizer(tokenizer_path)
    if tok.vocab_size > _MAX_UINT16_VOCAB:
        raise ValueError(
            f"vocab_size={tok.vocab_size} does not fit in uint16 — switch the shard dtype"
        )

    print(f"tokenizer: {tokenizer_path}  (vocab {tok.vocab_size})")
    print(f"indexing {len(paths)} corpus file(s)...")
    docs = _index_documents(paths)
    if not docs:
        raise ValueError("corpus contains no documents")

    # Shuffle whole documents, then split. See the module docstring for why the
    # order of these two operations is the entire point.
    random.Random(seed).shuffle(docs)
    n_val = max(1, int(len(docs) * val_fraction))
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    print(f"{len(docs):,} documents -> {len(train_docs):,} train / {len(val_docs):,} val")

    os.makedirs(out_dir, exist_ok=True)
    handles = [open(p, "rb") for p in paths]
    per_file = [0] * len(paths)
    try:
        n_train = _write_shard(handles, train_docs, tok, os.path.join(out_dir, "train.bin"), per_file)
        n_val_tok = _write_shard(handles, val_docs, tok, os.path.join(out_dir, "val.bin"), per_file)
    finally:
        for h in handles:
            h.close()

    total = n_train + n_val_tok
    meta = {
        "tokenizer": tokenizer_path,
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_id,
        "seed": seed,
        "n_documents": len(docs),
        "train_tokens": n_train,
        "val_tokens": n_val_tok,
        "sources": {
            os.path.basename(p): {
                "bytes": os.path.getsize(p),
                "tokens": per_file[i],
                "share": round(per_file[i] / total, 4),
                "bytes_per_token": round(os.path.getsize(p) / max(per_file[i], 1), 2),
            }
            for i, p in enumerate(paths)
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{total/1e6:.1f}M tokens  ({n_train/1e6:.1f}M train, {n_val_tok/1e6:.2f}M val)")
    for name, s in meta["sources"].items():
        print(f"  {name:<16} {s['tokens']/1e6:7.1f}M  {s['share']*100:5.1f}%  "
              f"{s['bytes_per_token']:.2f} bytes/token")
    # A default eval pass is eval_iters * batch_size * block_size tokens. If the
    # validation set is smaller than that, every eval resamples the same tokens
    # several times over and the "val loss" is a noisier number than it looks.
    if n_val_tok < _MIN_VAL_TOKENS:
        print(f"  warning: val is {n_val_tok/1e6:.2f}M tokens, below the ~"
              f"{_MIN_VAL_TOKENS/1e6:.1f}M a default eval pass samples "
              f"— raise --val-fraction")
    # Chinchilla's rule of thumb is ~20 tokens per parameter for a compute-optimal
    # run. Printing it here is the cheapest possible reality check on the corpus.
    print(f"\nChinchilla-optimal model size for this corpus: ~{total/20/1e6:.1f}M parameters")
    return meta


def load_meta(data_dir: str) -> dict:
    """Read the shard metadata written by `prepare` (vocab size, token counts, mixture)."""
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run: python -m scripts.prepare_data")
    with open(path) as f:
        return json.load(f)


# --- token shards -> batches -------------------------------------------------


def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    """Return one (x, y) batch of token windows from `<split>.bin`.

    x = tokens[i : i+block_size]      (the context)
    y = tokens[i+1 : i+block_size+1]  (each position's *next* token — the target)

    The two differ by a single position, which is the whole of the language-modelling
    objective: every position in the window predicts the one after it, so a batch of
    `batch_size` windows yields `batch_size * block_size` training signals, not one.

    The memmap is re-opened every call. That looks wasteful and isn't: holding one
    open across a long run leaks, because numpy keeps every page it has touched
    alive for the lifetime of the object, and a training run touches all of them.
    """
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run: python -m scripts.prepare_data")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) <= block_size:
        raise ValueError(f"{split}.bin has {len(data)} tokens, need > block_size={block_size}")

    # -block_size-1 so that y's last index (i + block_size) is always in range.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # int64 because nn.Embedding indexes with longs; the uint16 saving is about
    # disk and page cache, and it is paid back the moment the window is materialized.
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        # Pinned memory can be DMA'd to the GPU without a staging copy, so the
        # transfer overlaps the previous step's compute. MPS shares memory with the
        # CPU and has no equivalent, so it gets the plain path.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

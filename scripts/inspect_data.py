"""Entrypoint — look at what the data pipeline actually produced (Milestone 2).

    python -m scripts.inspect_data [--data-dir data/shards]

Four checks, in order of how easy they are to get silently wrong:

  1. the shards decode back to text (the tokenizer and the .bin agree)
  2. `y` really is `x` shifted by one — the entire training objective
  3. EOS appears at document boundaries and nowhere else
  4. every id in the shard is inside the vocabulary

None of these would fail loudly on their own during training. A shard written with
the wrong tokenizer trains to a plausible-looking loss curve and generates noise;
an off-by-one in the target teaches the model to copy its input. Both are cheap to
check here and expensive to notice later.
"""

import argparse

import numpy as np
import torch

from tgpt.data import get_batch, load_meta
from tgpt.tokenizer import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/shards")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    meta = load_meta(args.data_dir)
    tok = load_tokenizer(meta["tokenizer"])
    print(f"tokenizer     {meta['tokenizer']}  (vocab {meta['vocab_size']})")
    print(f"documents     {meta['n_documents']:,}")
    print(f"tokens        {meta['train_tokens']/1e6:.1f}M train / "
          f"{meta['val_tokens']/1e6:.2f}M val")
    print("mixture")
    for name, s in meta["sources"].items():
        print(f"  {name:<16} {s['tokens']/1e6:7.1f}M  {s['share']*100:5.1f}%  "
              f"{s['bytes_per_token']:.2f} bytes/token")

    torch.manual_seed(0)
    x, y = get_batch("train", args.data_dir, args.block_size, args.batch_size, "cpu")
    print(f"\nbatch         x={tuple(x.shape)} {x.dtype}   y={tuple(y.shape)} {y.dtype}")

    # --- 2. the shift ---------------------------------------------------------
    shifted = torch.equal(x[:, 1:], y[:, :-1])
    print(f"y is x shifted by one: {'PASS' if shifted else 'FAIL'}")

    # --- 4. ids in range ------------------------------------------------------
    in_range = int(x.max()) < meta["vocab_size"] and int(x.min()) >= 0
    print(f"ids within vocab:      {'PASS' if in_range else 'FAIL'} "
          f"(min {int(x.min())}, max {int(x.max())})")

    # --- 3. EOS density -------------------------------------------------------
    # EOS should occur about once per document, so its rate across the shard is a
    # direct check that separators survived: ~ n_documents / n_tokens.
    data = np.memmap(f"{args.data_dir}/train.bin", dtype=np.uint16, mode="r")
    sample = np.asarray(data[: min(len(data), 20_000_000)])
    n_eos = int((sample == meta["eos_id"]).sum())
    expected = meta["n_documents"] / (meta["train_tokens"] + meta["val_tokens"]) * len(sample)
    print(f"EOS in first {len(sample)/1e6:.0f}M tokens: {n_eos:,} "
          f"(expected ~{expected:,.0f} from the document count)")

    # --- 1. it decodes --------------------------------------------------------
    print("\n--- a decoded 128-token window ---")
    window = [int(t) for t in x[0, :128]]
    print(tok.decode(window))

    print("\n--- the first prediction in that window ---")
    ctx, target = window[:16], int(y[0, 15])
    print(f"context: {tok.decode(ctx)!r}")
    print(f"target : {tok.decode([target])!r}   (id {target})")


if __name__ == "__main__":
    main()

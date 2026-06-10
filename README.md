# tgpt — tiny GPT

A small GPT-2-style language model, **pretrained from scratch on Wikipedia**, built end-to-end
as a learning project. Every professional step is here for real: a SentencePiece subword
tokenizer, a sharded binary data pipeline, a GPT-2 architecture, and a proper training loop
(bf16, gradient accumulation, gradient clipping, weight-decay groups, warmup + cosine LR,
checkpoint/resume, evaluation, logging).

The point isn't a great model — it's a **real** one. `tgpt` is tiny and trains on a laptop, so
the text it generates is rough. But it does the one thing pretraining gives you for free:
**text completion**. Give it a prefix, it continues in Wikipedia-flavored English.

> Status: **scaffold** — the structure and interfaces are in place; the milestones below are
> implemented one at a time. This README will grow with results as each milestone lands.

## What it demonstrates

- **Subword tokenization** with SentencePiece (BPE), trained on the corpus itself.
- A **GPT-2 baseline** architecture (learned position embeddings, LayerNorm, GELU, biases,
  weight-tied head), then progressively **modernized** (RoPE, RMSNorm, SwiGLU, QK-norm, no-bias,
  Muon optimizer) as labeled, measurable upgrades.
- The **full pretraining pipeline** the pros use, scaled down to fit one machine.
- **Generative text completion** as the showcase capability.

## Architecture

Decoder-only transformer (GPT-2 family). Defaults are a "tiny" config sized for an Apple-Silicon
laptop; the same code scales to a GPT-2-small (124M) config on a rented GPU.

| | tiny (laptop default) | GPT-2 small (cloud target) |
|---|---|---|
| layers | 6 | 12 |
| heads | 6 | 12 |
| embedding dim | 384 | 768 |
| context length | 256 | 1024 |
| vocab (SentencePiece) | 16k | 32k–50k |

Modern toggles (off by default = faithful GPT-2 baseline; flipped on in Milestone 5):
`pos_embedding=rope`, `norm_type=rmsnorm`, `mlp_type=swiglu`, `qk_norm=true`, `bias=false`.

## Milestone roadmap

Each milestone is a set of `TODO(Mx)` blocks to implement. See [`docs/milestones.md`](docs/milestones.md).

- [ ] **M1 — Tokenizer** · train SentencePiece BPE; encode/decode round-trip — `tgpt/tokenizer.py`
- [ ] **M2 — Data pipeline** · corpus → tokenized `.bin` shards + memmap dataloader — `tgpt/data.py`
- [ ] **M3 — Model (GPT-2 baseline)** · attention, MLP, block, full model — `tgpt/model.py`
- [ ] **M4 — Training loop** · bf16, grad-accum, clip, AdamW groups, LR schedule, ckpt, eval — `tgpt/train.py`
- [ ] **M5 — Modernize** · RoPE, RMSNorm, SwiGLU, QK-norm, no-bias, Muon — toggles in `model.py`/`train.py`
- [ ] **M6 — Scale & generate** · bigger run, sampling, eval; the showcase — `tgpt/sample.py`

## Quickstart (once implemented)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# M1: train the tokenizer on the corpus
python -m scripts.train_tokenizer

# M2: tokenize + shard the corpus into binary files
python -m scripts.prepare_data

# M4/M6: train
python -m tgpt.train

# M6: generate a completion
python -m tgpt.sample --prompt "The history of"
```

## Hardware

Built and debugged on an Apple M3 Pro (18 GB unified memory, PyTorch + MPS). The tiny config
trains on the laptop; the GPT-2-small config is meant for a single rented GPU.

## License

MIT — see [`LICENSE`](LICENSE).

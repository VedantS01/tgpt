# tgpt milestones

Each milestone is a focused set of `TODO(Mx)` blocks in the code. Implement one at a time;
each should *run and produce visible output* before moving on. The theory for each is taught
alongside — this doc is the map, not the lesson.

The architecture starts as a faithful **GPT-2 baseline** and is then **modernized** so you can
measure what each modern choice buys.

---

## M1 — Tokenizer  ·  `tgpt/tokenizer.py`
Train a SentencePiece BPE tokenizer on the corpus; encode/decode round-trip.
- **Build:** `train_tokenizer`, `Tokenizer.__init__/encode/decode`.
- **Run:** `python -m scripts.train_tokenizer` → `tokenizer/tgpt.model`.
- **See:** a sentence split into subword pieces; `decode(encode(s)) == s`.
- **Learn:** subword vocabularies, BPE merges, vocab-size trade-offs, special tokens, byte-fallback.

## M2 — Data pipeline  ·  `tgpt/data.py`
Corpus → flat token array → `train.bin` / `val.bin` (uint16 memmap) → batched windows.
- **Build:** `download_corpus`, `prepare`, `get_batch`.
- **Run:** `python -m scripts.prepare_data` → `data/*.bin`.
- **See:** token counts; a sampled `(x, y)` batch where `y` is `x` shifted by one.
- **Learn:** memory-mapping, why uint16, document separators, train/val split, the next-token target.

## M3 — Model (GPT-2 baseline)  ·  `tgpt/model.py`
The faithful GPT-2 decoder: token + learned position embeddings, pre-norm blocks
(LayerNorm → attention → residual, LayerNorm → GELU-MLP → residual), final norm, weight-tied head.
- **Build:** `make_norm`, `CausalSelfAttention`, `MLP`, `Block`, `TGPT` (`__init__`, `forward`).
- **See:** parameter count; a forward pass producing `(B, T, vocab)` logits and a loss near `ln(vocab)`.
- **Learn:** multi-head attention, causal masking via SDPA, residual streams, pre-norm, weight tying, init.

## M4 — Training loop  ·  `tgpt/train.py`
The real pretraining engineering, none of which changes the model's parameters.
- **Build:** `get_lr`, `configure_optimizer`, `estimate_loss`, `save_checkpoint`, the step loop.
- **See:** train/val loss dropping; checkpoints written; resume working.
- **Learn:** bf16 autocast, gradient accumulation, gradient clipping, AdamW weight-decay groups,
  warmup + cosine LR, evaluation cadence, checkpointing, logging.

## M5 — Modernize  ·  toggles in `model.py` / `train.py`
Implement the modern option behind each `cfg` flag and A/B it against the baseline.
- **Build:** RMSNorm, RoPE, SwiGLU, QK-norm, no-bias; then the Muon optimizer branch.
- **See:** before/after val loss for each change (a small table you keep in the README).
- **Learn:** *why* each modern choice exists — and which ones actually matter at this scale.

## M6 — Scale & generate  ·  `tgpt/sample.py`
Put it together: a longer run, then sampling for the showcase.
- **Build:** `TGPT.generate`, `sample.load`, the `sample` CLI.
- **Run:** `python -m tgpt.sample --prompt "The history of"`.
- **See:** Wikipedia-flavored text completion (rough, but real — that's the demo).
- **Learn:** temperature, top-k, autoregressive decoding, honest evaluation of a tiny model.

---

### Cloud-later note
Everything is written to scale by swapping the config: bump `TGPTConfig` toward GPT-2-small
(12 layers / 12 heads / 768 dim / 1024 context), point `download_corpus` at the full
`wikimedia/wikipedia` dump (or FineWeb-Edu), and run the *same code* on a rented GPU.

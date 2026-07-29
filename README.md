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

- [x] **M1 — Tokenizer** · train SentencePiece BPE; encode/decode round-trip — `tgpt/tokenizer.py`
- [x] **M2 — Data pipeline** · corpus → tokenized `.bin` shards + memmap dataloader — `tgpt/data.py`
- [ ] **M3 — Model (GPT-2 baseline)** · attention, MLP, block, full model — `tgpt/model.py`
- [ ] **M4 — Training loop** · bf16, grad-accum, clip, AdamW groups, LR schedule, ckpt, eval — `tgpt/train.py`
- [ ] **M5 — Modernize** · RoPE, RMSNorm, SwiGLU, QK-norm, no-bias, Muon — toggles in `model.py`/`train.py`
- [ ] **M6 — Scale & generate** · bigger run, sampling, eval; the showcase — `tgpt/sample.py`

### M1 results

16k-piece BPE vocabulary trained on WikiText-103 (538M chars), with byte-fallback and
`unk=0, bos=1, eos=2` (pad disabled — GPT-style training windows are always full):

```
"The history of artificial intelligence began in antiquity."
  → ▁The ▁history ▁of ▁artificial ▁intelligence ▁began ▁in ▁antiqu ity .   (10 tokens)
"naïve café — résumé 😀 日本語"
  → round-trips exactly via byte-fallback pieces like <0xF0><0x9F><0x98><0x80>
```

Common words are single tokens; rare words split into meaningful chunks; anything outside
the learned vocabulary decomposes to bytes, so `decode(encode(s)) == s` always.

### M1b results — a code-aware tokenizer

A second tokenizer, byte-level BPE trained on 141 MB of Python + C++ + JS + prose, built to
a spec: whitespace ladder, operators and closers whole, all keywords whole, identifiers
deliberately split. Bytes per token on held-out files (higher is better):

| tokenizer | c++ | javascript | python | prose |
|---|---|---|---|---|
| M1 sentencepiece / wiki, 16k | 1.91 | 2.04 | 2.01 | **4.37** |
| M1b byte-level / code, 32k | **3.16** | **3.71** | **3.50** | 4.19 |
| | **+66%** | **+82%** | **+74%** | −4% |

Two things fell out of building it, both written up in
[`docs/tokenizer-design.md`](docs/tokenizer-design.md):

- **The M1 tokenizer was silently destroying whitespace.** `"a\nb"` decoded to `"a b"`,
  `"\n\n\n"` to `""`. Two SentencePiece defaults (`remove_extra_whitespaces`, and the
  `nmt_nfkc` normalizer rewriting control characters) — now fixed, with round-trip
  in the test set.
- **The requested 1/3/7/11/…/27 space ladder is exactly right, and 4n−1 is why.** A
  GPT-4-style regex stops indentation one space short because that space glues to the
  following word, so those rungs make any indent depth 1–7 a single token.

### M2 results — corpus to token shards

139 MB of Python, C++, JavaScript and Wikipedia prose, split into documents,
shuffled, and tokenized into `uint16` memmap shards:

| source | tokens | share | bytes/token |
|---|---|---|---|
| python.txt | 17.1M | 45.3% | 3.51 |
| prose.txt | 9.5M | 25.2% | 4.22 |
| javascript.txt | 7.3M | 19.3% | 3.44 |
| cpp.txt | 3.9M | 10.3% | 3.59 |
| **total** | **37.8M** | 36,078 documents | |

`python -m scripts.inspect_data` checks the four things that fail silently rather
than loudly — the shards decode back to text, `y` really is `x` shifted by one,
EOS lands at document boundaries (19,685 found vs 19,103 expected), and every id
is inside the vocabulary.

Two decisions are written up in [`docs/data-pipeline.md`](docs/data-pipeline.md):

- **Documents get shuffled before the train/val split, not after.** Each source is
  contiguous on disk, so slicing the tail of the token array would give you a
  validation set made of one language — and every later decision would be made on
  that number.
- **The corpus is 5× too small for the model it is meant to train.** 37.8M tokens
  is Chinchilla-optimal for 1.9M parameters; the tiny config is 10.6M. `prepare`
  prints that on every run rather than letting a training run discover it.

## Quickstart (once implemented)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# M1: train the tokenizer on the corpus
python -m scripts.train_tokenizer

# M2: tokenize + shard the corpus into binary files, then look at what came out
python -m scripts.prepare_data
python -m scripts.inspect_data

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

# tgpt — tiny GPT

A small GPT-2-style language model, **pretrained from scratch on code and Wikipedia**, built
end-to-end as a learning project. Every professional step is here for real: a purpose-built
subword tokenizer, a 600M-token corpus assembled from 16 open sources, a sharded binary data
pipeline, a GPT-2 architecture, and a proper training loop (bf16, gradient accumulation,
gradient clipping, weight-decay groups, warmup + cosine LR, checkpoint/resume, evaluation,
logging).

The point isn't a great model — it's a **real** one. `tgpt` is tiny and trains on a laptop, so
the text it generates is rough. But it does the one thing pretraining gives you for free:
**text completion**. Give it a prefix, it continues.

> Status: **scaffold** — the structure and interfaces are in place; the milestones below are
> implemented one at a time. This README will grow with results as each milestone lands.

## What it demonstrates

- **Subword tokenization** two ways: SentencePiece BPE, and a byte-level BPE whose
  pre-tokenization regex is designed against a spec — measurably better on both code and prose.
- **Corpus construction** as a first-class step: 16 openly-available sources, streamed to a
  byte budget, with the mixture as a readable registry.
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
| vocab | 32k (byte-level BPE) | 32k–50k |

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

A second tokenizer, byte-level BPE built to a spec: whitespace ladder, operators and
closers whole, all keywords whole, identifiers deliberately split. Bytes per token on
held-out text (higher is better), against the M1 SentencePiece baseline:

| held-out | php | javascript | python | c++ | wikipedia | wikihow | stackexchange | **weighted mean** |
|---|---|---|---|---|---|---|---|---|
| M1 sp / wikitext, 16k | 1.83 | 2.04 | 1.96 | 1.91 | 3.99 | 3.93 | 3.20 | 3.04 |
| M1b bpe / corpus, 32k | **3.68** | **3.73** | **3.51** | **3.14** | **4.25** | **5.00** | **4.00** | **4.00** |
| | +102% | +83% | +79% | +64% | +6% | +27% | +25% | **+32%** |

Written up in [`docs/tokenizer-design.md`](docs/tokenizer-design.md):

- **The M1 tokenizer was silently destroying whitespace.** `"a\nb"` decoded to `"a b"`,
  `"\n\n\n"` to `""`. Two SentencePiece defaults (`remove_extra_whitespaces`, and the
  `nmt_nfkc` normalizer rewriting control characters) — now fixed, with round-trip in
  the test set.
- **The requested 1/3/7/11/…/27 space ladder is exactly right, and 4n−1 is why.** A
  GPT-4-style regex stops indentation one space short because that space glues to the
  following word, so those rungs make any indent depth 1–7 a single token.
- **The prose regression turned out to be a corpus artifact, not a design cost.** The
  first version lost 4% on English. The same pre-tokenizer trained on the full corpus
  now beats the Wikipedia-trained baseline *on Wikipedia* — there was far more headroom
  in the corpus than in the configuration.

### M2 results — corpus and token shards

A 2.3 GB corpus across **16 sources** — ten programming languages, the English
Wikipedia dump, how-to and textbook prose, and StackExchange Q&A — declared as a
registry in `tgpt/corpus.py` and streamed to a byte budget:

| group | MB | share | | | tokens |
|---|---:|---:|---|---|---:|
| code | 1,050 | 45.6% | | wikipedia | 156.0M |
| wikipedia | 600 | 26.1% | | python | 113.6M |
| help / instructional | 500 | 21.7% | | everything else | 339.7M |
| Q&A | 150 | 6.5% | | **total** | **609.3M** |

609.3M tokens over 1,061,870 documents, shuffled and sharded into `uint16` memmaps
(605.0M train / 4.25M val). `python -m scripts.inspect_data` checks the four things
that fail silently rather than loudly — the shards decode back to text, `y` really is
`x` shifted by one, EOS lands at document boundaries (34,372 found vs 34,856
expected), and every id is inside the vocabulary.

Written up in [`docs/corpus.md`](docs/corpus.md) and
[`docs/data-pipeline.md`](docs/data-pipeline.md):

- **Documents get shuffled before the train/val split, not after.** Each source is
  contiguous on disk, so slicing the tail of the token array would give you a
  validation set made of one source — and every later decision would rest on it.
- **Availability is the hard part, not bandwidth.** Gated datasets, script-based
  datasets that `datasets` 5.0 removed, and one corpus that is entirely lowercased
  (dropped: the model is meant to be case-sensitive).
- **The corpus is now sized against the model.** 609M tokens is Chinchilla-optimal
  for 30.5M parameters, so the 10.6M tiny config trains at ~57 tokens/parameter. The
  previous corpus was optimal for 1.9M — `prepare` prints this on every run.

## Quickstart (once implemented)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# M1/M1b: build the corpus, then train the tokenizer on it
python -m scripts.fetch_corpus
python -m scripts.train_code_tokenizer

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

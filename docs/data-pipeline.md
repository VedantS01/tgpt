# Data pipeline — design notes (M2)

M1 turned text into token ids. M2 turns a corpus into the thing a training loop
can actually read from: a flat array of ids on disk, sampled as random windows.

```
corpus files (.txt, NUL-separated documents)
   -> index document offsets          (no text in RAM)
   -> shuffle documents, then split    (train / val)
   -> tokenize in batches, stream out  (train.bin / val.bin, uint16)
   -> get_batch: memmap -> (x, y)      (y = x shifted by one)
```

Three of these steps have a wrong version that still runs, trains, and produces a
loss curve. Those are the ones worth writing down.

## Documents are a unit, so the corpus records where they end

Every corpus file is a concatenation of documents separated by a NUL byte
(`tgpt.data.DOC_SEP`). NUL is the choice because it is the one byte that never
occurs in real text, so the split can never be ambiguous — no escaping, no
sentinel string that might appear inside a document.

It never reaches the tokenizer. `prepare` splits on it and puts an **EOS token**
at each boundary; `train_code_tokenizer` splits on it too, so BPE never spends a
merge on it and never learns a pair that straddles two unrelated files.

Without boundaries the model has no signal for "this text is finished", which is
precisely what generation needs in order to stop. It would also learn that a
Python file is routinely followed mid-thought by a C++ header.

Most sources give boundaries for free — one dataset record is one document. The
exception is WikiText-103, which ships as one line per paragraph with no article
markers except the headings themselves, so `download_corpus` detects top-level
` = Title = ` lines — exactly one `=` per side, since ` = = Section = = ` is the
same article — and cuts there, recovering **29,444 articles** from a flat stream.

## Shuffle first, split second

The corpus is a mixture of 16 sources, and each is contiguous on disk: all the
Python, then all the C++, then Wikipedia, then the how-to articles. Take the last 0.5%
of the *token array* as validation — the obvious thing, and what a single-source
pipeline gets away with — and the validation set is entirely whichever source
landed last. You would be measuring prose loss and calling it val loss, and every
architecture decision downstream would be made on that number.

So `prepare` pools documents from all sources, shuffles with a fixed seed, and
splits the shuffled list. Two properties follow:

- both splits hold the same mixture, so val loss means what it says
- the shuffle is at **document** granularity, so no document is half in train and
  half in val — a paragraph memorized from the training half cannot show up as
  suspiciously low loss on the validation half

The seed is recorded in `meta.json`, so the split is reproducible across re-runs.

## Validation has to be big enough to evaluate with, and no bigger

A default eval pass samples `eval_iters * batch_size * block_size` = 100 x 32 x 256
= **0.8M tokens**. A validation set smaller than that is resampled several times per
evaluation, and the "val loss" is noisier than its decimal places suggest.

A plain fraction is wrong at both ends. 0.5% of a 140 MB corpus is 0.17M tokens —
too small to evaluate on. 4% of a 2.3 GB corpus is 24M tokens — training data thrown
away for no extra signal. So `val_fraction` is a share of corpus *bytes*, clamped to
4-16 MB, which lands at roughly 1M-4M validation tokens at any corpus size. The
2.3 GB corpus holds out 16 MB across 7,763 documents, giving 4.25M val tokens.

## uint16, and re-opening the memmap

Ids fit in `uint16` while `vocab_size < 65,536` — ours is 32,000 — which halves
disk and page-cache pressure against `int32`. `prepare` refuses to write if the
vocabulary would not fit, rather than letting numpy wrap silently.

`get_batch` re-opens the memmap on every call. That looks wasteful and isn't:
a memmap object keeps every page it has touched alive, and a training run touches
all of them, so a long-lived handle turns into the whole corpus resident in
memory — the exact thing memmap exists to avoid.

## The target is the input, shifted by one

```
x = tokens[i     : i+block_size]
y = tokens[i + 1 : i+block_size+1]
```

Every position in the window predicts the next one, so a batch of 32 windows of
256 tokens is **8,192** training signals, not 32. This is why language modelling
gets so much out of unlabeled text, and it is an off-by-one away from teaching the
model to copy its input instead — hence the explicit check in
`scripts.inspect_data`.

## What came out

2.3 GB across 16 sources (see [`docs/corpus.md`](corpus.md)), tokenized with the
M1b 32k code tokenizer:

| source | tokens | share | bytes/token |
|---|---:|---:|---:|
| wikipedia | 156.0M | 25.6% | 3.86 |
| python | 113.6M | 18.6% | 3.52 |
| stackexchange | 44.6M | 7.3% | 3.38 |
| web_edu | 44.0M | 7.2% | 4.34 |
| python_github | 35.2M | 5.8% | 3.41 |
| javascript | 33.4M | 5.5% | 3.55 |
| wikihow | 32.2M | 5.3% | 4.98 |
| code_qa | 24.9M | 4.1% | 4.02 |
| go | 20.6M | 3.4% | 2.94 |
| java | 20.1M | 3.3% | 3.99 |
| openstax | 18.6M | 3.0% | 4.85 |
| cpp | 18.3M | 3.0% | 3.26 |
| php | 15.7M | 2.6% | 3.83 |
| khanacademy | 14.8M | 2.4% | 4.05 |
| rosetta | 10.8M | 1.8% | 2.81 |
| ruby | 6.3M | 1.0% | 3.42 |
| **total** | **609.3M** | 1,061,870 docs | 3.78 |

605.0M train / 4.25M val. EOS appears 34,372 times in the first 20M tokens against
34,856 expected from the document count — the boundaries survived. A decoded window
comes back as clean prose or as real code with its indentation intact, which is the
round-trip check that would have caught the M1 whitespace bug.

## Chinchilla, and how the corpus was sized

Chinchilla's rule of thumb is ~20 tokens per parameter, and `prepare` prints the
implied model size on every run rather than leaving it to be discovered after a
training run that plateaus early.

The first corpus was 37.8M tokens — compute-optimal for a **1.9M**-parameter model,
against a tiny config of **10.6M**. That is the gap that motivated the 2.3 GB
rebuild. At 609.3M tokens the corpus is now compute-optimal for **30.5M**
parameters, so the tiny config sits at ~57 tokens per parameter: comfortably
over-trained, which is what you want for a small model — modern practice trains
small models far past Chinchilla because inference cost, not training cost, is what
they are optimized for.

It also means the corpus no longer caps the project. Scaling toward GPT-2-small
(124M parameters, ~2.5B tokens) is `--total-mb` and patience, not a redesign.

## Shards are keyed to a tokenizer

`meta.json` records which tokenizer produced a shard set, and training reads it
instead of trusting a hardcoded `vocab_size` — a model built with the wrong
vocabulary trains to a plausible-looking curve and generates noise.

Building a second set under a different tokenizer is how the M1-vs-M1b comparison
becomes measurable at the loss level rather than only in bytes per token:

```bash
python -m scripts.prepare_data                                     # -> data/shards
python -m scripts.prepare_data --tokenizer tokenizer/tgpt.model \
    --out data/shards-sp                                           # -> data/shards-sp
```

Both read the same corpus and the same seed, so the document split is identical
and only the tokenizer differs. Note that this doubles disk: 609M tokens is 1.2 GB
per set.

## Scale

The 2.3 GB / 1.06M-document run indexes, shuffles, tokenizes and writes in a few
minutes. Peak memory is set by the batch of documents in flight (1,024 at a time),
not by the corpus, so the same code runs unchanged on a corpus that does not fit
in RAM — which is the entire reason for the offset-index-then-stream design rather
than the obvious "read it all, tokenize it, write it".

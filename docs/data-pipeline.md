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

Recovering them cost something. WikiText-103 ships as one line per paragraph with
no article markers except the headings themselves, so `download_corpus` detects
top-level ` = Title = ` lines — exactly one `=` per side, since ` = = Section = = `
is the same article — and cuts there. That turns a flat stream of paragraphs back
into **29,444 articles**.

## Shuffle first, split second

The corpus is a mixture, and each source is contiguous on disk: all the Python,
then all the C++, then all the JavaScript, then all the prose. Take the last 0.5%
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

## Validation has to be big enough to evaluate with

`val_fraction` is a share of **documents**, not tokens, and the default is 4%.
That is much larger than the 0.05% a big-corpus pipeline would use, for an
arithmetic reason: a default eval pass samples `eval_iters * batch_size *
block_size` = 100 × 32 × 256 ≈ **0.8M tokens**. A validation set smaller than that
is being resampled several times per evaluation, and the "val loss" is noisier
than its decimal places suggest. `prepare` prints a warning when it happens.

At 37.8M tokens, 4% of documents is 1.28M val tokens. On a corpus ten times the
size, lower it.

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

139 MB of Python, C++, JavaScript and Wikipedia prose, tokenized with the M1b
32k code tokenizer:

| source | tokens | share | bytes/token |
|---|---|---|---|
| python.txt | 17.1M | 45.3% | 3.51 |
| prose.txt | 9.5M | 25.2% | 4.22 |
| javascript.txt | 7.3M | 19.3% | 3.44 |
| cpp.txt | 3.9M | 10.3% | 3.59 |
| **total** | **37.8M** | | |

36,078 documents → 36.5M train / 1.28M val. EOS appears 19,685 times in the first
20M tokens against 19,103 expected from the document count — the boundaries
survived. A decoded window comes back as real Python with its indentation intact,
which is the round-trip check that would have caught the M1 whitespace bug.

## The gap this exposes

Chinchilla's rule of thumb is ~20 tokens per parameter. 37.8M tokens is
compute-optimal for a **1.9M-parameter** model, and tgpt's tiny config is
**10.6M** — so this corpus is roughly 5× too small to train it properly.
`prepare` prints that number on every run rather than leaving it to be discovered
after a training run that plateaus early.

It is a corpus problem, not a pipeline problem, and closing it is one flag:

```bash
# add the full 539 MB WikiText corpus alongside the code
python -m scripts.prepare_data --sources 'data/code/*.txt' data/corpus.txt
```

Scaling the *code* side is a matter of raising the per-language byte targets in
`scripts/fetch_code_corpus.py`; the sources stream, so only the target changes.
The spec's remaining model-corpus sources — StackOverflow, Medium, research
papers, wikiHow, instruction manuals — are additional fetchers of the same shape:
yield documents, and `_write_stream` handles the rest.

## Two shard sets, on purpose

Shards are keyed to a tokenizer, so building both makes the M1-vs-M1b comparison
measurable at the loss level rather than only in bytes per token:

```bash
python -m scripts.prepare_data                                   # -> data/shards
python -m scripts.prepare_data --sources data/corpus.txt \
    --tokenizer tokenizer/tgpt.model --out data/shards-wiki      # -> data/shards-wiki
```

| shard set | corpus | tokenizer | tokens | bytes/token |
|---|---|---|---|---|
| `data/shards` | 139 MB code + prose | M1b code, 32k | 37.8M | 3.68 |
| `data/shards-wiki` | 539 MB WikiText-103 | M1 sentencepiece, 16k | 123.8M | 4.36 |

`meta.json` records which tokenizer produced each set, and training reads it
instead of trusting a hardcoded `vocab_size` — a model built with the wrong
vocabulary trains to a plausible-looking curve and generates noise.

The 539 MB run is also the scale test: one file, 29,444 documents, tokenized and
sharded in about a minute, with peak memory set by the batch of documents in
flight rather than by the corpus. EOS lands 4,725 times in the first 20M tokens
against 4,756 expected.

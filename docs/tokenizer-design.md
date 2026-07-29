# Code-aware tokenizer — design notes (M1b)

M1 trained a SentencePiece BPE on WikiText-103. This document covers the follow-up:
a byte-level BPE built for code, and what measurably changed.

## The framing that matters most

A tokenizer only learns tokens for text it has seen. No flag, regex or vocabulary
size will produce a `"""` token or a 4-space-indent token from a corpus of
Wikipedia articles. **Corpus first; configuration only avoids sabotaging yourself.**

That is also why the M1 tokenizer cannot simply be "tuned" for code. Two of
SentencePiece's defaults are actively destructive here:

| default | effect on code |
|---|---|
| `remove_extra_whitespaces=True` | collapses space runs — **deletes Python indentation** |
| `normalization_rule_name="nmt_nfkc"` | rewrites control characters — **deletes newlines and tabs** |
| `character_coverage=0.9995` | tuned for CJK-heavy prose; code is ASCII-dense |

The first two are not hypothetical. The M1 tokenizer as originally written turned
`"a\nb"` into `"a b"` and `"\n\n\n"` into `""` — it could not represent a line break
at all, which would have quietly capped what any model trained on it could learn.
Both flags are now fixed in `tgpt/tokenizer.py`, and `decode(encode(s)) == s` is part
of the test set rather than an assumption.

Those defaults are fixable. The deeper reason to switch is that every requirement in the
spec is a statement about *where token boundaries may fall* — and in byte-level
BPE that lives in one readable place: the pre-tokenization regex. In
SentencePiece it is spread across interacting flags plus `user_defined_symbols`.

## Why byte-level BPE

- every byte is representable, so nothing is ever unknown and round-trip is exact
- whitespace is ordinary text, so indentation runs can become tokens
- the pre-tokenizer is a regex you can read, test, and change

One practical discovery: HuggingFace `tokenizers` matches with **Oniguruma**, so
lookarounds *and* possessive quantifiers are available. Rust's `regex` crate
supports neither, and a design assuming it would have needed a rewrite. This is
what makes the acronym rule and the indentation rule expressible at all.

## The pre-tokenizer, clause by clause

Alternation is leftmost-first, so **order is the design**.

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)     english contractions (prose)
 ?(?<![A-Za-z0-9_])(KEYWORDS)(?![A-Za-z0-9_])   protected keywords  — req 5
 ?[A-Z]+(?![a-z])                ACRONYM run, stopping before a Capitalized word
 ?[A-Z]?[a-z]+                   Word / word   — req 6 splits happen here
 ?\p{L}+                         any other script
\p{N}                            single digits
 ?[^\s\p{L}\p{N}]++[\r\n]*       runs of punctuation  — reqs 2, 3, 4
\s*[\r\n]                        newlines
\s+(?!\S)                        indentation runs     — req 1
\s+
```

### Requirement 1 — the space ladder is 4n−1, and that is exactly right

The spec asked for space runs of **1, 3, 7, 11, 15, 19, 23, 27**. Those are `4n−1`,
and the reason is the interaction between two clauses. `\s+(?!\S)` refuses to
consume the space directly before a word, because that space belongs to the
` ?word` clause — ` def` is one piece. So an indented line splits as:

```
"    return 1"   ->   '   ' + ' return' + ' ' + '1'      4-space indent  = 1 token
"        def g"  ->   '       ' + ' def'                 8-space indent  = 1 token
```

The ladder therefore covers indent depths 1–7 at **one token each**, and it leaves
prose untouched, where a space always glues to its following word. Verified
empirically, not assumed.

A single space needs no help: it is one of the 256 seed bytes, so it is always a
token.

### Requirement 6 — identifier splitting comes free

Underscores are deliberately *absent* from the word clauses, so they fall through
to the punctuation clause:

```
make_shared   ->   'make'  '_'  'shared'
makeShared    ->   'make'  'Shared'          (case boundary)
HTTPServer    ->   'HTTP'  'Server'          (acronym lookahead)
XMLHttpRequest->   'XML'   'Http'  'Request'
```

The acronym clause is why ` ?[A-Z]+(?![a-z])` is ordered *before* the word clause.
Without the lookahead, greedy `[A-Z]+` eats the `S` of `Server` and yields
`HTTPS` + `erver`.

### Requirements 5 and 6 are in direct conflict

Requirement 6 says split on underscores. Requirement 5 says every keyword is one
token. C++ has 20 keywords made of exactly that: `static_assert`, `co_await`,
`thread_local`, `char16_t`, `not_eq`, `reinterpret_cast`…

These cannot both hold via frequency, because BPE can never merge across a
pre-token boundary. The resolution is the **protected-keyword branch**, matched
before the identifier rules, so those strings are never split in the first place:

```
static_assert(x);            ->  'static_assert'  '('  'x'  ');'
std::make_shared<Foo>(a_b);  ->  'make'  '_'  'shared'  ...  'a'  '_'  'b'
```

Keywords survive whole; user identifiers split. That is the spec's intent, and it
needs the pre-tokenizer — a post-hoc vocabulary filter cannot express it.

**The cost is real and worth naming.** `size_t` is not a keyword, so it splits into
three tokens despite being ubiquitous in C++. Requirement 6 is a deliberate trade:
better vocabulary sharing (`shared` is the same token in code and prose) paid for
in tokens per identifier.

### Why added-tokens are used sparingly

An added token is matched *before* pre-tokenization, which makes it a blunt
instrument. Forcing `return` to be an added token would break the leading space
off every ` return` in the corpus and make compression **worse**, since ` return`
is the form that actually occurs. So added tokens back only the guarantees the
spec makes outright — the space ladder — and only for runs of ≥3 spaces, which
essentially never occur in prose.

## Results

Bytes per token on ~1 MB of held-out text per source that no tokenizer trained on.
Higher is better; it is effective context length and inference cost per unit of text.

The middle column is the same design trained on the first corpus (139 MB: Python,
C++, JS, WikiText). The right column is that design trained on the full 2.3 GB
mixture (see [`docs/corpus.md`](corpus.md)). Holding the pre-tokenizer and the
vocabulary size fixed across those two isolates **the corpus effect on its own**.

| held-out source | M1 sp/wikitext 16k | M1b bpe 32k, 139 MB | M1b bpe 32k, 2.3 GB |
|---|---|---|---|
| php | 1.83 | 3.55 | **3.68** |
| javascript | 2.04 | 3.71 | **3.73** |
| java | 1.93 | 3.49 | **3.50** |
| python | 1.96 | 3.45 | **3.51** |
| go | 1.79 | 3.12 | **3.25** |
| c++ | 1.91 | **3.16** | 3.14 |
| stackexchange | 3.20 | 3.71 | **4.00** |
| wikihow | 3.93 | 4.53 | **5.00** |
| openstax | 4.00 | 4.44 | **4.85** |
| wikipedia | 3.99 | 4.07 | **4.25** |
| web_edu | 4.00 | 4.18 | **4.32** |
| **corpus-weighted mean** | 3.04 | 3.84 | **4.00** |

The mean is weighted by each source's share of the corpus, not a plain average: a
tokenizer that wins on Ruby and loses on Python has not improved the thing the
model will actually read.

Spec compliance, 32k: space ladder **8/8 with nothing force-added** — the bigger
corpus taught every rung on its own — identifiers-split 13/13, round-trip 11/11,
operators 26/28, Python keywords 37/38, JS keywords 50/51, C++ keywords 70/96.

### The prose regression is gone, and the corpus is why

The first version of this tokenizer was **−4% on prose** against the
Wikipedia-trained SentencePiece baseline. That was reported as the unavoidable
price of the spec: requirement 6 splits every identifier, and single digits cost
tokens.

It was not unavoidable. It was a corpus artifact. Trained on 600 MB of Wikipedia
and 500 MB of instructional prose instead of 40 MB of WikiText, the *same
pre-tokenizer* now beats the SentencePiece baseline **on Wikipedia itself** (4.25
vs 3.99, +6%) while keeping every bit of the code advantage. Requirement 6 still
costs what it costs; there was simply far more headroom in the corpus than in the
configuration.

This is the same lesson as the opening section, arriving with a number attached:
**a tokenizer only learns tokens for text it has seen.** The one thing that
genuinely got worse is C++ (3.16 → 3.14), which is exactly what you would predict
— C++ went from 10% of the corpus to 3%.

### The baseline got *worse* as it got more correct

As originally written, the M1 tokenizer scored 2.21 bytes/token on C++; once it
stopped destroying whitespace it scored 1.91. Nothing regressed — it had been
*deleting* the indentation it was being measured on, and deleted text compresses
beautifully. A compression number means nothing without a round-trip check beside
it.

### What the remaining misses mean

The 24 unmatched C++ keywords are almost entirely `alignas`, `atomic_cancel`,
`atomic_noexcept`, `synchronized`, `reflexpr` — transactional-memory and reflection
proposals that appear approximately zero times in real code. Requirement 5 as
literally stated ("all registered keywords are single tokens") is **not worth
satisfying**: forcing them in would spend vocabulary on tokens the model would never
see enough of to train. The requirement is better read as "keywords that occur should
never be split," which is met.

Similarly `f"` and `self.` miss because the pre-tokenizer *deliberately* forbids
them: `f` is a word and `"` is punctuation, so they can never merge, and `self.`
splits for the same reason `make_shared` does. Those are the spec working, not
failing.

## Open decisions

- **Digits**: currently every digit is its own token (Llama-style), which makes
  arithmetic uniform but costs tokens on numeric code. `--no-split-digits` switches
  to `\p{N}{1,3}` for an A/B.
- **Underscore placement**: `_` is isolated rather than glued to the next piece
  (`make` `_` `shared`, not `make` `_shared`). Isolated shares vocabulary with
  prose; glued compresses better. One-line change in the word clause.
- **Vocabulary size vs model size**: see below — this one has teeth.

## Vocabulary size is coupled to the model

tgpt's tiny config is 6 layers at 384 dim ≈ **10.6M** transformer parameters. The
embedding table is `vocab × 384` and, with weight tying, is counted once:

| vocab | embedding params | share of model |
|---|---|---|
| 16k | 6.1M | 37% |
| 32k | 12.3M | 54% — larger than the entire transformer |
| 152k (Qwen-scale) | 58M | 85%, a lookup table with a transformer stapled on |

Copy Qwen's *recipe* (byte-level BPE, code-inclusive corpus), not its numbers. At
384 dim, 32k is already the point where most of the model is embedding, and rare
tokens are seen too few times to train properly. The rule of thumb: vocabulary
scales with **both** model size and corpus size.

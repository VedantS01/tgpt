"""Milestone 1b — a code-aware byte-level BPE tokenizer.

M1 trained a SentencePiece BPE on Wikipedia. That tokenizer is fine for prose and
bad for code, for a reason no flag can fix: SentencePiece's defaults collapse runs
of whitespace (deleting Python indentation outright), and a Wikipedia corpus
contains almost no triple-quotes, `});` or 4-space indents to learn from in the
first place.

This module is the alternative: byte-level BPE (the GPT-2/GPT-4 lineage) built on
HuggingFace `tokenizers`, where the *pre-tokenization regex* is the knob. That
regex decides what BPE is even allowed to merge, which makes it the single place
where "handle whitespace, quotes, operators and identifiers well" actually lives.

Why byte-level rather than SentencePiece:
  - every byte is representable, so nothing is ever unknown and round-trip is exact
  - whitespace is ordinary text, so indentation runs can become tokens
  - the pre-tokenizer is a readable regex instead of a pile of interacting flags

The interface matches `tgpt.tokenizer.Tokenizer` (vocab_size / encode / decode) so
the rest of the pipeline does not care which one it is loaded with.
"""

from __future__ import annotations

import os
import re

from . import langspec


# --- the pre-tokenization regex ---------------------------------------------
# Read top to bottom: alternation is leftmost-first, so ORDER IS THE DESIGN.
# HuggingFace `tokenizers` matches with Oniguruma, so lookarounds and possessive
# quantifiers are both available (Rust's `regex` crate would support neither).


def build_pattern(split_digits: bool = True, protect: list[str] | None = None) -> str:
    """Build the pre-tokenization pattern.

    Each branch maps to a requirement:

      protected keywords  req 5, and the escape hatch from req 6 (see below)
      ACRONYM run         `HTTPServer` -> `HTTP` + `Server`, not `HTTPS` + `erver`
      Word / word         req 6: camelCase and PascalCase split at the case boundary
      digits              uniform numeric behaviour (no `1234`-is-a-token weirdness)
      preprocessor        `#include` / `#define`, which `#` + word could never merge into
      punctuation run     reqs 2, 3, 4: `);`, `===`, `\"\"\"`, `${` merge into one piece
      newline             kept distinct from horizontal whitespace
      indentation run     req 1: `\\s+(?!\\S)` yields runs of 4n-1 spaces

    Underscores are deliberately absent from the word branches, so `make_shared`
    falls apart into `make` + `_` + `shared` (req 6) with no extra machinery.
    """
    protect = langspec.protected_keywords() if protect is None else protect
    # Longest-first so `static_assert` wins over any shorter prefix, and bounded by
    # identifier-character lookarounds so we only ever match a whole word.
    alts = "|".join(re.escape(k) for k in sorted(protect, key=len, reverse=True))
    protected = rf"| ?(?<![A-Za-z0-9_])(?:{alts})(?![A-Za-z0-9_])" if protect else ""
    digits = r"|\p{N}" if split_digits else r"|\p{N}{1,3}"

    return (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"     # english contractions (prose)
        + protected                          # req 5 (and the documented req-6 exception)
        + r"| ?[A-Z]+(?![a-z])"              # ACRONYM, stopping before a Capitalized word
        r"| ?[A-Z]?[a-z]+"                   # Word / word  -- req 6 splits happen here
        r"| ?\p{L}+"                         # any other script
        + digits
        + r"| ?\#[a-z]+"                     # C preprocessor: #include, #define, #pragma
        + r"| ?[^\s\p{L}\p{N}]++[\r\n]*"     # runs of punctuation -- reqs 2, 3, 4
        r"|\s*[\r\n]"                        # newlines
        r"|\s+(?!\S)"                        # indentation runs   -- req 1
        r"|\s+"
    )


def build_pre_tokenizer(split_digits: bool = True, protect: list[str] | None = None):
    from tokenizers import Regex, pre_tokenizers

    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex(build_pattern(split_digits, protect)), behavior="isolated", invert=False
        ),
        # Map bytes -> printable stand-ins. use_regex=False because the Split above
        # has already decided the pieces; ByteLevel must not re-split them.
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


# Special ids match M1 exactly, so both tokenizers are drop-in for the pipeline.
SPECIALS = ["<unk>", "<bos>", "<eos>"]
UNK_ID, BOS_ID, EOS_ID = 0, 1, 2


def _documents(corpus_files: list[str], caps: dict[str, float] | None = None,
               block: int = 1 << 22):
    """Stream `corpus_files` as documents, splitting on the corpus separator byte.

    `caps` optionally limits how many bytes to read from each file, which is how
    tokenizer training stays tractable on a corpus far larger than it needs.
    """
    from .data import DOC_SEP

    for path in corpus_files:
        cap = (caps or {}).get(path, float("inf"))
        buf, n = "", 0
        with open(path, encoding="utf-8", errors="replace") as f:
            while n < cap and (chunk := f.read(block)):
                n += len(chunk)
                buf += chunk
                *done, buf = buf.split(DOC_SEP)
                yield from (d for d in done if d)
        if buf:
            yield buf


def sample_caps(corpus_files: list[str], sample_bytes: float | None) -> dict[str, float]:
    """Per-file byte caps that sample `sample_bytes` total, proportional to file size.

    Proportional rather than a flat per-file cap: the tokenizer should see the
    corpus's actual mixture. A flat cap would give a 40 MB Ruby source the same
    weight as a 400 MB Python one and quietly retune the vocabulary away from what
    the model will actually be trained on.
    """
    sizes = {p: os.path.getsize(p) for p in corpus_files}
    total = sum(sizes.values())
    if not sample_bytes or total <= sample_bytes:
        return {}
    frac = sample_bytes / total
    return {p: s * frac for p, s in sizes.items()}


def train_code_tokenizer(
    corpus_files: list[str],
    out_path: str,
    vocab_size: int = 32000,
    split_digits: bool = True,
    min_frequency: int = 2,
    sample_bytes: float | None = 300e6,
) -> str:
    """Train a byte-level BPE over `corpus_files` and save it to `out_path`."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = build_pre_tokenizer(split_digits)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIALS,
        # Seed the full 256-byte alphabet so no byte is ever unrepresentable.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # Feed documents rather than raw files: the corpus stores NUL between documents
    # (see tgpt.data.DOC_SEP) and the tokenizer should never see that byte, or it
    # would spend merges learning it. Splitting here also stops BPE from learning
    # pairs that straddle two unrelated files.
    #
    # A tokenizer needs representative statistics, not volume — the merge ranking
    # for a 32k vocabulary is settled long before 300 MB — so the corpus is
    # subsampled proportionally rather than read whole.
    caps = sample_caps(corpus_files, sample_bytes)
    if caps:
        print(f"sampling {sum(caps.values())/1e6:.0f} MB of "
              f"{sum(os.path.getsize(p) for p in corpus_files)/1e6:.0f} MB for training")
    tok.train_from_iterator(_documents(corpus_files, caps), trainer)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    return out_path


class CodeTokenizer:
    """Byte-level BPE tokenizer. Same interface as `tgpt.tokenizer.Tokenizer`."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — train it first (scripts/train_code_tokenizer.py)."
            )
        from tokenizers import Tokenizer as HFTokenizer

        self.tok = HFTokenizer.from_file(model_path)
        self.bos_id, self.eos_id = BOS_ID, EOS_ID

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(
        self, texts: list[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        """Encode many texts at once — the Rust backend threads this internally."""
        batch = [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]
        if add_bos:
            batch = [[self.bos_id] + ids for ids in batch]
        if add_eos:
            batch = [ids + [self.eos_id] for ids in batch]
        return batch

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # --- helpers used by the audit ------------------------------------------
    def n_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def pieces(self, text: str) -> list[str]:
        return self.tok.encode(text, add_special_tokens=False).tokens


def ensure_single_tokens(model_path: str, targets: list[str]) -> list[str]:
    """Force `targets` to be single tokens, returning the ones actually added.

    Used only as a backstop for the guarantees the spec makes outright (the space
    ladder, req 1). Applied sparingly on purpose: an added token is matched before
    pre-tokenization, so adding something like `return` would break the leading
    space off every ` return` in the corpus and make compression *worse*.
    """
    from tokenizers import AddedToken, Tokenizer as HFTokenizer

    tok = HFTokenizer.from_file(model_path)
    missing = [
        t for t in targets
        if len(tok.encode(t, add_special_tokens=False).ids) != 1
    ]
    if missing:
        tok.add_tokens([AddedToken(t, normalized=False, special=False) for t in missing])
        tok.save(model_path)
    return missing

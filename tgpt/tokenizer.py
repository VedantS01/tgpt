"""Milestone 1 — Subword tokenizer (SentencePiece BPE).

A tokenizer maps text <-> a sequence of integer ids. Unlike the character-level toy
tokenizers, this one learns *subword* units (frequent character chunks) from the corpus,
so common words become single tokens and rare words split into pieces. That is what lets a
fixed, modest vocabulary cover open-ended text — and it is the unit the model predicts.

We use Google's SentencePiece (it both TRAINS and RUNS a tokenizer, unlike tiktoken which
is inference-only). Train it once on the corpus; the result is a `.model` file you load
everywhere else (data pipeline, sampling).

Implement the three pieces below. The rest of the project depends only on this interface:
    Tokenizer.train(...)  -> writes <prefix>.model / <prefix>.vocab
    tok = Tokenizer(model_path)
    tok.encode("hello") -> list[int]
    tok.decode([1,2,3]) -> "hello"
"""

from __future__ import annotations
import os

# Special-token ids, fixed here so every stage of the pipeline agrees on them.
# GPT-style models don't pad (training windows are always full), so pad is disabled.
# <unk> must exist even with byte_fallback — SentencePiece requires it — but byte
# fallback means it essentially never fires: unknown characters decompose into the
# 256 byte tokens instead of collapsing into a lossy <unk>.
PAD_ID = -1   # disabled
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2    # doubles as the document separator in the data pipeline (M2)


def train_tokenizer(
    corpus_path: str,
    model_prefix: str,
    vocab_size: int = 16000,
    model_type: str = "bpe",          # "bpe" matches the GPT lineage; "unigram" is SP's default
    character_coverage: float = 0.9995,
) -> str:
    """Train a SentencePiece model on a plain-text corpus and return the .model path."""
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=True,
        # SentencePiece defaults to remove_extra_whitespaces=True, which silently
        # collapses every whitespace run: "a\nb" round-trips as "a b" and "\n\n\n"
        # as "". That erases line and paragraph structure the model should be
        # learning, and would delete indentation outright on any code. Both flags
        # below are required for decode(encode(s)) == s to hold on real text.
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        # ...and even with those two, the default `nmt_nfkc` normalizer still
        # rewrites control characters, so tabs and newlines never survive. Only
        # `identity` leaves the bytes alone, which is what makes round-trip exact.
        normalization_rule_name="identity",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        # BPE training holds its working set in RAM, so subsample: 10M sentences
        # drawn uniformly from the corpus is far more than enough signal for a
        # 16k-merge vocabulary, and keeps training fast on a laptop.
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
    )
    return f"{model_prefix}.model"


class Tokenizer:
    """Thin wrapper around a trained SentencePiece model."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — train it first (scripts/train_tokenizer.py)."
            )
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text -> list of token ids."""
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(
        self, texts: list[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        """Encode many texts at once — SentencePiece releases the GIL and threads it."""
        batch = self.sp.encode(texts, out_type=int, num_threads=os.cpu_count())
        if add_bos:
            batch = [[self.bos_id] + ids for ids in batch]
        if add_eos:
            batch = [ids + [self.eos_id] for ids in batch]
        return batch

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text (special tokens are dropped by SentencePiece)."""
        return self.sp.decode(ids)


def load_tokenizer(path: str):
    """Load whichever tokenizer `path` points at.

    The project has two: the M1 SentencePiece model (`.model`) and the M1b
    code-aware byte-level BPE (`.json`). They expose the same interface on
    purpose, so everything downstream — the data pipeline, training, sampling —
    only needs to be told a path, never which family it belongs to.
    """
    if path.endswith(".json"):
        from .code_tokenizer import CodeTokenizer

        return CodeTokenizer(path)
    return Tokenizer(path)

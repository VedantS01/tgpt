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


def train_tokenizer(
    corpus_path: str,
    model_prefix: str,
    vocab_size: int = 16000,
    model_type: str = "bpe",          # "bpe" matches the GPT lineage; "unigram" is SP's default
    character_coverage: float = 0.9995,
) -> str:
    """Train a SentencePiece model on a plain-text corpus and return the .model path.

    TODO(M1):
      - import sentencepiece as spm
      - call spm.SentencePieceTrainer.train(...) with:
            input=corpus_path, model_prefix=model_prefix, vocab_size=vocab_size,
            model_type=model_type, character_coverage=character_coverage,
            byte_fallback=True,          # any unknown char decomposes to bytes (no <unk> loss)
            pad_id=..., unk_id=..., bos_id=..., eos_id=...,   # decide your special-token ids
        Consider input_sentence_size / shuffle_input_sentence for very large corpora
        (SentencePiece subsamples internally — you do not need the whole corpus to train).
      - return f"{model_prefix}.model"

    Returns the path to the trained .model file.
    """
    # TODO(M1): implement
    raise NotImplementedError("M1: train the SentencePiece model")


class Tokenizer:
    """Thin wrapper around a trained SentencePiece model."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — train it first (scripts/train_tokenizer.py)."
            )
        # TODO(M1): load the model
        #   import sentencepiece as spm
        #   self.sp = spm.SentencePieceProcessor(model_file=model_path)
        raise NotImplementedError("M1: load the SentencePiece model")

    @property
    def vocab_size(self) -> int:
        # TODO(M1): return self.sp.get_piece_size()
        raise NotImplementedError

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text -> list of token ids.

        TODO(M1): return self.sp.encode(text, out_type=int) and, if requested, prepend
        bos / append eos. (Document separators matter for the data pipeline in M2.)
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text.

        TODO(M1): return self.sp.decode(ids).
        """
        raise NotImplementedError

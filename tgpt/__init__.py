"""tgpt — a tiny GPT-2-style language model pretrained from scratch.

Package layout:
    config.py     model + training configuration dataclasses
    tokenizer.py  SentencePiece subword tokenizer (Milestone 1)
    data.py       corpus -> token shards + dataloader (Milestone 2)
    model.py      GPT-2 baseline architecture, modern toggles (Milestones 3 & 5)
    train.py      the professional training loop (Milestone 4)
    sample.py     load a checkpoint and generate text (Milestone 6)
"""

__version__ = "0.0.1"

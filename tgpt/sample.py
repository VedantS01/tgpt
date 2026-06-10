"""Milestone 6 — load a trained checkpoint and generate text (the showcase).

Pretraining's free gift is text COMPLETION: give the model a prefix, it predicts a plausible
continuation. That's all this does — load weights, encode a prompt, autoregressively sample,
decode. It is the thing you'll demo.

Run:  python -m tgpt.sample --prompt "The history of"
"""

from __future__ import annotations
import argparse
import torch

from .model import TGPT
from .tokenizer import Tokenizer


def load(ckpt_path: str, tokenizer_path: str, device: str):
    """Rebuild the model from a checkpoint and load the tokenizer.

    TODO(M6):
      - ckpt = torch.load(ckpt_path, weights_only=False)
      - model = TGPT(ckpt["model_cfg"]).to(device); model.load_state_dict(ckpt["model"]); model.eval()
      - tok = Tokenizer(tokenizer_path)
      - return model, tok
    """
    raise NotImplementedError("M6: load checkpoint + tokenizer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--ckpt", type=str, default="checkpoints/tgpt.pt")
    p.add_argument("--tokenizer", type=str, default="tokenizer/tgpt.model")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # TODO(M6):
    #   - model, tok = load(args.ckpt, args.tokenizer, device)
    #   - ids = tok.encode(args.prompt) or a start token if empty
    #   - idx = torch.tensor([ids], device=device)
    #   - out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    #   - print(tok.decode(out[0].tolist()))
    raise NotImplementedError("M6: wire up prompt -> generate -> decode")


if __name__ == "__main__":
    main()

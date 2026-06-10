"""Milestone 4 — the professional training loop (with Milestone 5 optimizer swap).

This is where the real pretraining engineering lives. Implement each numbered piece. None of
it changes the model's parameters — it's all about driving the optimization well and being able
to stop/resume/evaluate like a real run.

Run:  python -m tgpt.train
"""

from __future__ import annotations
import os
import math
import torch

from .config import TGPTConfig, TrainConfig
from .model import TGPT
from .data import get_batch


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Learning-rate schedule: linear warmup then cosine decay to min_lr.

    TODO(M4):
      - step < warmup_steps:  linear ramp 0 -> learning_rate
      - warmup..max_steps:    cosine decay learning_rate -> min_lr
      - after max_steps:      min_lr
    """
    raise NotImplementedError("M4: implement the LR schedule")


def configure_optimizer(model: TGPT, cfg: TrainConfig):
    """Build the optimizer with correct weight-decay groups.

    The professional detail: apply weight decay ONLY to matmul weights (2D tensors). Do NOT
    decay biases, norm weights, or embeddings — decaying those hurts.

    TODO(M4):
      - split params into decay (ndim >= 2) and no_decay (ndim < 2) groups
      - torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(beta1, beta2))
        with weight_decay=cfg.weight_decay on the decay group, 0.0 on the other
    TODO(M5):
      - if cfg.optimizer == "muon": use Muon on the 2D hidden weights and AdamW on the rest
        (embeddings, head, norms, biases). Measure the difference vs plain AdamW.
    """
    raise NotImplementedError("M4: build optimizer with weight-decay groups")


@torch.no_grad()
def estimate_loss(model, model_cfg, train_cfg, device):
    """Average loss over a few batches of train and val. Returns {"train":..., "val":...}.

    TODO(M4):
      - model.eval(); for each split, average model(*get_batch(...))[1] over eval_iters; model.train()
    """
    raise NotImplementedError("M4: implement eval loss estimate")


def save_checkpoint(path, model, optimizer, step, model_cfg, train_cfg):
    """Save everything needed to resume AND to rebuild the model standalone.

    TODO(M4): torch.save({model, optimizer, step, model_cfg, train_cfg, rng_state}, path)
    """
    raise NotImplementedError("M4: implement checkpoint save")


def train(model_cfg: TGPTConfig | None = None, train_cfg: TrainConfig | None = None):
    model_cfg = model_cfg or TGPTConfig()
    train_cfg = train_cfg or TrainConfig()
    device = resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)

    model = TGPT(model_cfg).to(device)
    optimizer = configure_optimizer(model, train_cfg)

    # The full loop. Implement the marked steps.
    for step in range(train_cfg.max_steps + 1):
        # TODO(M4): set this step's LR on every optimizer param group via get_lr(step, train_cfg)

        # TODO(M4): periodic eval (every eval_every): call estimate_loss, print/log train+val
        # TODO(M4): periodic checkpoint (every ckpt_every): save_checkpoint(...)

        # --- one optimizer step with gradient accumulation ---
        # TODO(M4):
        #   optimizer.zero_grad(set_to_none=True)
        #   for micro in range(grad_accum_steps):
        #       xb, yb = get_batch("train", train_cfg.data_dir, model_cfg.block_size,
        #                          train_cfg.batch_size, device)
        #       with torch.autocast(device_type=..., dtype=bf16/fp16):   # mixed precision
        #           _, loss = model(xb, yb)
        #       (loss / grad_accum_steps).backward()                     # scale for accumulation
        #   torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)  # clip
        #   optimizer.step()
        #   (if fp16: use a GradScaler)
        # TODO(M4): lightweight logging every log_every (loss, lr, grad-norm, tokens/sec)
        raise NotImplementedError("M4: implement the training step")


if __name__ == "__main__":
    train()

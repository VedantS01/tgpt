"""Configuration for tgpt.

Two dataclasses, deliberately separated so the distinction is obvious:

  - TGPTConfig  : the MODEL's shape and architecture choices. Anything that changes
                  the network's parameters or forward pass lives here. It is saved
                  inside every checkpoint so a model can always be rebuilt exactly.

  - TrainConfig : the TRAINING run's knobs (batch size, learning rate, schedule, ...).
                  These do NOT change the model's parameters; you can resume the same
                  weights under a different TrainConfig.

The defaults below describe the "tiny" laptop model. The commented GPT-2-small numbers
are the cloud-scale target — the same code should run both by swapping the config.
"""

from dataclasses import dataclass


@dataclass
class TGPTConfig:
    # --- vocabulary (must match the trained SentencePiece model, Milestone 1) ---
    vocab_size: int = 16000          # GPT-2 small uses 50257; we use a smaller subword vocab

    # --- model shape (tiny default; GPT-2 small = 12 / 12 / 768 / 1024) ---
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256            # context length (tokens of history the model sees)

    # --- regularization ---
    dropout: float = 0.0
    bias: bool = True                # GPT-2 uses biases in Linear/Norm; Milestone 5 flips to False

    # --- weight tying: share the token-embedding matrix with the output head (GPT-2 does this) ---
    weight_tying: bool = True

    # --- modern architecture toggles (Milestone 5) ------------------------------
    # All default to the GPT-2 BASELINE choice. M5 implements the modern option and
    # you flip these on to measure the effect of each one.
    pos_embedding: str = "learned"   # "learned" (GPT-2)  |  "rope" (modern, M5)
    norm_type: str = "layernorm"     # "layernorm" (GPT-2)|  "rmsnorm" (modern, M5)
    mlp_type: str = "gelu"           # "gelu" (GPT-2)      |  "swiglu" (modern, M5)
    qk_norm: bool = False            # normalize Q,K before attention (modern, M5)

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    # --- data ---
    data_dir: str = "data"           # where the .bin token shards live (Milestone 2)

    # --- optimization ---
    batch_size: int = 32             # sequences per micro-step
    grad_accum_steps: int = 8        # micro-steps per optimizer step -> effective batch = batch_size * grad_accum_steps
    max_steps: int = 5000            # number of OPTIMIZER steps
    learning_rate: float = 3e-4      # peak LR (after warmup)
    min_lr: float = 3e-5             # final LR (end of cosine decay)
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0           # global-norm gradient clipping
    beta1: float = 0.9
    beta2: float = 0.95

    # --- optimizer choice (Milestone 5) ---
    optimizer: str = "adamw"         # "adamw" (baseline) | "muon" (modern, M5)

    # --- precision ---
    dtype: str = "bf16"              # "bf16" | "fp16" | "fp32"  (bf16 preferred on MPS/modern GPUs)

    # --- evaluation + checkpointing ---
    eval_every: int = 250            # run validation every N optimizer steps
    eval_iters: int = 100            # number of val batches to average for the loss estimate
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1000

    # --- logging ---
    log_every: int = 10
    wandb_project: str = ""          # "" disables wandb; set a project name to enable (M4, optional)

    # --- system ---
    device: str = "auto"             # "auto" -> mps if available else cpu
    seed: int = 1337

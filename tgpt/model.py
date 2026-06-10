"""Milestones 3 & 5 — the model.

M3 builds the faithful GPT-2 BASELINE: token + learned position embeddings, a stack of
pre-norm transformer blocks (LayerNorm -> attention -> residual, LayerNorm -> MLP(GELU) ->
residual), a final norm, and a weight-tied output head.

M5 turns each `cfg` toggle into a measurable upgrade: RoPE positions, RMSNorm, SwiGLU MLP,
QK-norm, and dropping biases. Implement the modern branch behind the toggle so you can A/B it.

Implement each module's __init__ and forward where marked. Keep the forward shapes:
    idx : (B, T) int64 token ids
    returns logits (B, T, vocab_size) and, if targets given, a scalar cross-entropy loss.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import TGPTConfig


def make_norm(cfg: TGPTConfig) -> nn.Module:
    """Return the normalization layer for this config.

    TODO(M3): return nn.LayerNorm(cfg.n_embd, bias=cfg.bias) for the baseline.
    TODO(M5): if cfg.norm_type == "rmsnorm", return an RMSNorm module instead
              (RMSNorm divides by the root-mean-square; no mean subtraction, no bias).
    """
    raise NotImplementedError("M3: return LayerNorm (M5: add RMSNorm branch)")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, cfg: TGPTConfig):
        super().__init__()
        # TODO(M3):
        #   - a projection from n_embd -> 3*n_embd for q,k,v (one Linear is conventional),
        #     respecting cfg.bias
        #   - an output projection n_embd -> n_embd
        #   - attn / residual dropout (cfg.dropout)
        #   - stash n_head, head_dim, block_size
        # TODO(M5):
        #   - if cfg.qk_norm: a norm applied to q and k per-head before attention
        #   - if cfg.pos_embedding == "rope": precompute/buffer the rotary frequencies
        raise NotImplementedError("M3: build the attention projections")

    def forward(self, x):
        # x: (B, T, C)
        # TODO(M3):
        #   - project to q, k, v; reshape each to (B, n_head, T, head_dim)
        #   - F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)  # the fused, masked path
        #   - merge heads back to (B, T, C); output projection + dropout
        # TODO(M5):
        #   - apply qk_norm to q,k if enabled
        #   - apply RoPE rotation to q,k if cfg.pos_embedding == "rope"
        raise NotImplementedError("M3: implement attention forward")


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, cfg: TGPTConfig):
        super().__init__()
        # TODO(M3): GPT-2 MLP — Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd) -> dropout
        # TODO(M5): if cfg.mlp_type == "swiglu" — gated variant:
        #           two input projections (gate, up), SiLU(gate) * up, then down-projection.
        #           Size the hidden dim so the param count stays comparable (~2/3 * 4 * n_embd).
        raise NotImplementedError("M3: build the MLP")

    def forward(self, x):
        # TODO(M3 / M5): implement the chosen MLP forward
        raise NotImplementedError("M3: implement MLP forward")


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, each with a residual."""

    def __init__(self, cfg: TGPTConfig):
        super().__init__()
        # TODO(M3): norm1, attention, norm2, mlp  (use make_norm(cfg))
        raise NotImplementedError("M3: build the block")

    def forward(self, x):
        # TODO(M3): x = x + attn(norm1(x)); x = x + mlp(norm2(x)); return x
        raise NotImplementedError("M3: implement block forward")


class TGPT(nn.Module):
    """The full GPT-2-style decoder."""

    def __init__(self, cfg: TGPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(M3):
        #   - token embedding: nn.Embedding(vocab_size, n_embd)
        #   - position embedding: nn.Embedding(block_size, n_embd)   [baseline; skip if RoPE in M5]
        #   - dropout
        #   - nn.ModuleList of cfg.n_layer Blocks
        #   - final norm (make_norm)
        #   - lm_head: Linear(n_embd -> vocab_size, bias=False)
        #   - if cfg.weight_tying: self.lm_head.weight = self.token_emb.weight
        #   - initialize weights (GPT-2 uses normal(0, 0.02); scale residual-projection inits by
        #     1/sqrt(2*n_layer) — worth implementing and understanding).
        raise NotImplementedError("M3: build the full model")

    def forward(self, idx, targets=None):
        # idx: (B, T)
        # TODO(M3):
        #   - token + position embeddings (position only for the learned-embedding baseline)
        #   - run through blocks, final norm, lm_head -> logits (B, T, vocab_size)
        #   - if targets is not None: loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        #   - return logits, loss
        raise NotImplementedError("M3: implement model forward")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend `idx` (B, T) by max_new_tokens. Used by sampling (M6).

        TODO(M6):
          - loop: crop idx to the last block_size tokens; forward; take logits[:, -1, :];
            apply temperature; optional top-k filtering; softmax; multinomial sample; append.
        """
        raise NotImplementedError("M6: implement generation")

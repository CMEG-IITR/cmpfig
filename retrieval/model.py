"""
Dual-encoder model: CLIP vision tower + MatBERT text encoder.
Both project to a shared `embed_dim`-dimensional space and are L2-normalised.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel

from .config import Config


class DualEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        _attn  = "flash_attention_2" if config.use_flash_attn else "eager"
        _dtype = torch.bfloat16  # load directly in bfloat16 — avoids float32→bf16 cast

        # Image encoder: CLIP vision tower (ViT-B/32 → 768-dim pooler)
        self.image_encoder = CLIPVisionModel.from_pretrained(
            config.image_encoder, attn_implementation=_attn, torch_dtype=_dtype,
        )
        img_dim = self.image_encoder.config.hidden_size  # 768

        # Text encoder: MatBERT (BERT-base → 768-dim CLS)
        self.text_encoder = AutoModel.from_pretrained(
            config.text_encoder, attn_implementation=_attn, torch_dtype=_dtype,
        )
        txt_dim = self.text_encoder.config.hidden_size  # 768

        # Projection heads: linear (no bias) + independent LayerNorm
        self.image_proj = nn.Sequential(
            nn.Linear(img_dim, config.embed_dim, bias=False),
            nn.LayerNorm(config.embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(txt_dim, config.embed_dim, bias=False),
            nn.LayerNorm(config.embed_dim),
        )

        # Learnable log-temperature (initialised to log(0.07))
        if config.learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(config.init_tau)))
        else:
            self.register_buffer(
                "log_tau", torch.tensor(math.log(config.init_tau))
            )

    # ── Encoding helpers (used standalone during eval) ──────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised image embeddings [B, embed_dim]."""
        out = self.image_encoder(pixel_values=pixel_values)
        # pooler_output = post_layernorm(CLS token)  shape: [B, 768]
        feat = out.pooler_output
        return F.normalize(self.image_proj(feat), dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns L2-normalised text embeddings [B, embed_dim]."""
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token at position 0
        feat = out.last_hidden_state[:, 0]
        return F.normalize(self.text_proj(feat), dim=-1)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=0.01, max=100.0)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, pixel_values, input_ids, attention_mask):
        img_emb = self.encode_image(pixel_values)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb, self.tau


# ── Optimiser parameter groups ────────────────────────────────────────────────

def build_param_groups(model: DualEncoder, config: Config):
    """
    Three LR tiers:
      lr_vision  — CLIP vision encoder (general web images → needs shift for matscience)
      lr_text    — MatBERT text encoder (already matscience domain → light adjustment)
      lr_proj    — projection heads + learnable temperature
    Bias / LayerNorm weights excluded from weight decay in all groups.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    vis_wd,  vis_nd  = [], []
    txt_wd,  txt_nd  = [], []
    proj_wd, proj_nd = [], []

    for mod, wd_list, nd_list in [
        (model.image_encoder, vis_wd,  vis_nd),
        (model.text_encoder,  txt_wd,  txt_nd),
        (model.image_proj,    proj_wd, proj_nd),
        (model.text_proj,     proj_wd, proj_nd),
    ]:
        for name, param in mod.named_parameters():
            if not param.requires_grad:
                continue
            (nd_list if any(nd in name for nd in no_decay) else wd_list).append(param)

    proj_wd.append(model.log_tau)

    return [
        {"params": vis_wd,  "lr": config.lr_vision, "weight_decay": config.weight_decay},
        {"params": vis_nd,  "lr": config.lr_vision, "weight_decay": 0.0},
        {"params": txt_wd,  "lr": config.lr_text,   "weight_decay": config.weight_decay},
        {"params": txt_nd,  "lr": config.lr_text,   "weight_decay": 0.0},
        {"params": proj_wd, "lr": config.lr_proj,   "weight_decay": config.weight_decay},
        {"params": proj_nd, "lr": config.lr_proj,   "weight_decay": 0.0},
    ]

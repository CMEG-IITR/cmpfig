from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_root: str = "/mnt/d/Subham/Compoun_img_01/main_data"
    dataset_index: str = "dataset_index.json"
    output_dir: str = "/mnt/d/Subham/Compoun_img_01/retrieval/outputs"

    # ── Split (80/10/10 of total panels via figure-level integrity) ─────────
    train_frac: float = 0.80
    val_frac: float   = 0.10
    test_frac: float  = 0.10
    rare_subtype_threshold: int = 10   # subtypes with < N figures → always train
    split_seed: int = 42

    # ── Encoders ───────────────────────────────────────────────────────────
    image_encoder: str = "openai/clip-vit-base-patch32"
    text_encoder: str  = "m3rg-iitd/matscibert"
    embed_dim: int = 512
    max_text_len: int = 256

    # ── Temperature ────────────────────────────────────────────────────────
    init_tau: float = 0.07
    learnable_tau: bool = True

    # ── Optimiser ──────────────────────────────────────────────────────────
    lr_vision: float   = 3e-5   # CLIP — never seen materials science images, needs more shift
    lr_text: float     = 2e-5   # MatBERT — already trained on matscience text, light adjustment
    lr_proj: float     = 1e-4   # projection heads + temperature
    adam_eps: float    = 1e-6   # safer than 1e-8 under bfloat16
    weight_decay: float = 0.01
    warmup_fraction: float = 0.10
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 256
    num_epochs: int = 15
    # Two-level hard negative sampling:
    #   Level 1: same subtype, different figure
    #   Level 2: if subtype pool < hard_neg_fallback_threshold → same category
    hard_neg_ratio: float = 0.5
    hard_neg_fallback_threshold: int = 50   # min subtype pool size for level-1

    # ── Evaluation ─────────────────────────────────────────────────────────
    recall_k: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])
    mrr_cutoff: int = 1000         # search depth for MRR (rank > cutoff ≈ 0)
    min_category_test_samples: int = 200

    # ── Logging / Checkpointing ────────────────────────────────────────────
    log_every: int = 100
    eval_every: int = 1000         # more steps/epoch with 313K panels
    save_every: int = 3000

    # ── Runtime ────────────────────────────────────────────────────────────
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42
    use_flash_attn: bool = False  # flash-attn ABI often mismatches; keep False unless rebuilt from source

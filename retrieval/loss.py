"""
Symmetric InfoNCE (CLIP-style) contrastive loss.
"""
import torch
import torch.nn.functional as F


def infonce_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric cross-entropy over the [B x B] cosine-similarity logit matrix.
    img_emb, txt_emb: [B, D], already L2-normalised.
    tau: scalar temperature (learnable or fixed).

    The diagonal contains the B true (image, text) pairs; all off-diagonal
    entries are negatives.  Hard negatives constructed by the batch sampler
    naturally appear in the off-diagonal cells.
    """
    B = img_emb.size(0)
    logits = (img_emb @ txt_emb.T) / tau          # [B, B]
    labels = torch.arange(B, device=img_emb.device)

    loss_i2t = F.cross_entropy(logits,   labels)   # each row  = image query
    loss_t2i = F.cross_entropy(logits.T, labels)   # each col  = text  query

    return (loss_i2t + loss_t2i) / 2

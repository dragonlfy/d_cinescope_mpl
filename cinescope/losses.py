from typing import Dict
import torch
import torch.nn.functional as F


def compute_losses(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], lambda_attr: float = 1.0, lambda_ot: float = 0.2, lambda_delta: float = 1e-4) -> Dict[str, torch.Tensor]:
    score_loss = F.mse_loss(outputs["score"], targets["score"])
    attr_loss = F.mse_loss(outputs["attributes"], targets["attributes"])
    total = score_loss + lambda_attr * attr_loss + lambda_ot * outputs["ot_loss"] + lambda_delta * outputs["delta_l2"]
    return {"total": total, "score": score_loss.detach(), "attr": attr_loss.detach(), "ot": outputs["ot_loss"].detach(), "delta": outputs["delta_l2"].detach()}

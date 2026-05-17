from typing import Dict
import numpy as np
import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = pred.detach().cpu().numpy().reshape(-1)
    y = target.detach().cpu().numpy().reshape(-1)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_score_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {"MAE": mae(pred, target), "RMSE": rmse(pred, target), "PLCC": pearson_corr(pred, target)}

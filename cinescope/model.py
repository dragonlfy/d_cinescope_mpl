from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTRIBUTE_NAMES


@dataclass
class ModelConfig:
    image_size: int = 224
    dim: int = 128
    patch_size: int = 16
    num_prompt_tokens: int = 4
    sinkhorn_iters: int = 20
    epsilon: float = 0.05
    beta_prior: float = 0.1


class ToyVisualEncoder(nn.Module):
    """Small patch encoder used for a runnable reference implementation."""

    def __init__(self, dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        feat = feat.flatten(2).transpose(1, 2)
        feat = self.norm(feat)
        feat = feat + self.mlp(feat)
        return feat


class HCPDPromptTree(nn.Module):
    """Hierarchical Cinematographic Prompt Decomposition."""

    def __init__(self, dim: int = 128, num_prompt_tokens: int = 4, sub_attributes: Dict[str, List[str]] = None):
        super().__init__()
        self.dim = dim
        self.num_prompt_tokens = num_prompt_tokens
        self.dim_names = ATTRIBUTE_NAMES
        if sub_attributes is None:
            sub_attributes = {
                "composition": ["rule_of_thirds", "symmetry", "leading_lines"],
                "lighting": ["key_light", "contrast", "shadow"],
                "color": ["harmony", "palette", "temperature"],
                "motion": ["subject_motion", "camera_direction", "pacing"],
                "narrative": ["intent", "emotion", "continuity"],
            }
        self.sub_attributes = sub_attributes
        self.num_dims = len(self.dim_names)
        self.num_sub = max(len(v) for v in sub_attributes.values())
        self.root = nn.Parameter(torch.randn(dim) * 0.02)
        self.dim_contexts = nn.Parameter(torch.randn(self.num_dims, dim) * 0.02)
        self.sub_contexts = nn.Parameter(torch.randn(self.num_dims, self.num_sub, dim) * 0.02)
        self.synthesizers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(self.num_dims)
        ])
        self.delta = nn.Parameter(torch.randn(self.num_dims, self.num_sub, num_prompt_tokens, dim) * 0.01)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_list, dim_ids = [], []
        for l in range(self.num_dims):
            for k in range(self.num_sub):
                base = self.synthesizers[l](torch.cat([self.root, self.dim_contexts[l], self.sub_contexts[l, k]], dim=-1))
                tokens = base[None, :] + self.delta[l, k]
                prompt_list.append(tokens)
                dim_ids.extend([l] * self.num_prompt_tokens)
        prompts = torch.cat(prompt_list, dim=0)
        prompt_dim_ids = torch.tensor(dim_ids, device=prompts.device, dtype=torch.long)
        return prompts, prompt_dim_ids

    def delta_l2(self) -> torch.Tensor:
        return (self.delta ** 2).mean()


class SinkhornOT(nn.Module):
    """Entropy-regularized OT for patch-prompt alignment."""

    def __init__(self, epsilon: float = 0.05, iters: int = 20, beta_prior: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.iters = iters
        self.beta_prior = beta_prior

    def forward(self, visual: torch.Tensor, prompts: torch.Tensor, visual_dim_logits: torch.Tensor, prompt_dim_ids: torch.Tensor):
        B, Nv, _ = visual.shape
        Nt = prompts.shape[0]
        Kdim = visual_dim_logits.shape[-1]
        visual_n = F.normalize(visual, dim=-1)
        prompts_n = F.normalize(prompts, dim=-1)
        cost = 1.0 - torch.einsum("bnd,td->bnt", visual_n, prompts_n)
        visual_pi = F.softmax(visual_dim_logits, dim=-1)
        prompt_onehot = F.one_hot(prompt_dim_ids, Kdim).float()
        prior_sim = torch.einsum("bnk,tk->bnt", visual_pi, prompt_onehot)
        cost = cost + self.beta_prior * (1.0 - prior_sim)
        u = torch.full((B, Nv), 1.0 / Nv, device=visual.device)
        v = torch.full((B, Nt), 1.0 / Nt, device=visual.device)
        Kmat = torch.exp(-cost / max(self.epsilon, 1e-6)).clamp_min(1e-12)
        a = torch.ones_like(u)
        b = torch.ones_like(v)
        for _ in range(self.iters):
            a = u / (torch.einsum("bnt,bt->bn", Kmat, b) + 1e-8)
            b = v / (torch.einsum("bnt,bn->bt", Kmat, a) + 1e-8)
        T = a[:, :, None] * Kmat * b[:, None, :]
        ot_loss = (T * cost).sum(dim=(1, 2)).mean()
        return T, cost, ot_loss


class CineScopeMPL(nn.Module):
    """Simplified CineScope-MPL model."""

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.dim_names = ATTRIBUTE_NAMES
        self.num_dims = len(self.dim_names)
        self.visual_encoder = ToyVisualEncoder(dim=cfg.dim, patch_size=cfg.patch_size)
        self.hcpd = HCPDPromptTree(dim=cfg.dim, num_prompt_tokens=cfg.num_prompt_tokens)
        self.visual_dim_head = nn.Linear(cfg.dim, self.num_dims)
        self.ot = SinkhornOT(cfg.epsilon, cfg.sinkhorn_iters, cfg.beta_prior)
        self.evidence_norm = nn.LayerNorm(cfg.dim)
        self.score_head = nn.Sequential(nn.Linear(cfg.dim * self.num_dims, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        self.attr_head = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))

    def aggregate_evidence(self, T: torch.Tensor, visual: torch.Tensor, prompt_dim_ids: torch.Tensor) -> torch.Tensor:
        evidence = []
        for k in range(self.num_dims):
            mask = (prompt_dim_ids == k).float()
            weights = (T * mask[None, None, :]).sum(dim=-1)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            ev = torch.einsum("bn,bnd->bd", weights, visual)
            evidence.append(ev)
        return self.evidence_norm(torch.stack(evidence, dim=1))

    def forward(self, images: torch.Tensor):
        visual = self.visual_encoder(images)
        prompts, prompt_dim_ids = self.hcpd()
        visual_dim_logits = self.visual_dim_head(visual)
        T, cost, ot_loss = self.ot(visual, prompts, visual_dim_logits, prompt_dim_ids)
        evidence = self.aggregate_evidence(T, visual, prompt_dim_ids)
        score = torch.sigmoid(self.score_head(evidence.flatten(1))).squeeze(-1)
        attrs = torch.sigmoid(self.attr_head(evidence).squeeze(-1))
        return {
            "score": score,
            "attributes": attrs,
            "evidence": evidence,
            "transport": T,
            "cost": cost,
            "ot_loss": ot_loss,
            "delta_l2": self.hcpd.delta_l2(),
        }


def build_model_from_dict(cfg_dict: Dict) -> CineScopeMPL:
    return CineScopeMPL(ModelConfig(**cfg_dict))

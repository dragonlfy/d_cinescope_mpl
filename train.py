import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml
from cinescope.data import StoryboardDataset
from cinescope.losses import compute_losses
from cinescope.metrics import summarize_score_metrics
from cinescope.model import build_model_from_dict


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        image = batch["image"].to(device)
        score = batch["score"].to(device)
        out = model(image)
        preds.append(out["score"].cpu())
        targets.append(score.cpu())
    return summarize_score_metrics(torch.cat(preds), torch.cat(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cinescope_toy.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/toy_run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = StoryboardDataset(args.data_root, args.ann_file, image_size=cfg["model"]["image_size"])
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"].get("num_workers", 0))
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers", 0))
    model = build_model_from_dict(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            image = batch["image"].to(device)
            target = {"score": batch["score"].to(device), "attributes": batch["attributes"].to(device)}
            out = model(image)
            losses = compute_losses(out, target, cfg["train"]["lambda_attr"], cfg["train"]["lambda_ot"], cfg["train"]["lambda_delta"])
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            pbar.set_postfix(total=f"{float(losses['total']):.4f}", score=f"{float(losses['score']):.4f}", attr=f"{float(losses['attr']):.4f}", ot=f"{float(losses['ot']):.4f}")
        metrics = evaluate(model, val_loader, device)
        print(f"[val] epoch={epoch} metrics={metrics}")
        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": metrics}, output_dir / "best.pt")
            print(f"Saved best checkpoint to {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

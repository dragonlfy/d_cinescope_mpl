import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from cinescope.feedback import attrs_to_dict, generate_feedback
from cinescope.model import build_model_from_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model_from_dict(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    image_size = cfg["model"]["image_size"]
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    image = Image.open(args.image).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    score = float(out["score"][0].cpu())
    attr_dict = attrs_to_dict(out["attributes"][0].cpu().tolist())
    result = {"image": str(Path(args.image)), "score": round(score, 4), "attributes": {k: round(v, 4) for k, v in attr_dict.items()}, "feedback": generate_feedback(score, attr_dict)}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

ATTRS = ["composition", "lighting", "color", "motion", "narrative"]


def draw_sample(path: Path, attrs):
    W, H = 224, 224
    img = Image.new("RGB", (W, H), (245, 245, 235))
    draw = ImageDraw.Draw(img)
    c = int(80 + attrs["color"] * 150)
    img.paste((c, 180, 220), [0, 0, W, H])
    x = int(W * (0.33 + (1 - attrs["composition"]) * random.uniform(-0.25, 0.35)))
    y = int(H * (0.45 + random.uniform(-0.1, 0.1)))
    r = 14
    light = int(80 + attrs["lighting"] * 170)
    draw.polygon([(0, 0), (W, 0), (x, y)], fill=(light, light, 120))
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(30, 30, 30))
    if attrs["motion"] > 0.45:
        dx = int(30 + attrs["motion"] * 40)
        draw.line((x, y + 35, x + dx, y + 35), fill=(20, 80, 20), width=4)
        draw.polygon([(x + dx, y + 35), (x + dx - 8, y + 29), (x + dx - 8, y + 41)], fill=(20, 80, 20))
    if attrs["narrative"] > 0.5:
        draw.rectangle((150, 145, 200, 190), outline=(80, 40, 40), width=3)
        draw.line((150, 145, 200, 190), fill=(80, 40, 40), width=2)
    draw.rectangle((4, 4, W - 5, H - 5), outline=(20, 20, 20), width=2)
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / "annotations.jsonl"
    rng = random.Random(7)
    with ann_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            attrs = {name: rng.random() for name in ATTRS}
            score = 0.85 * (sum(attrs.values()) / len(attrs)) + 0.15 * rng.random()
            img_name = f"sample_{i:04d}.png"
            draw_sample(img_dir / img_name, attrs)
            item = {"image": f"images/{img_name}", "score": round(score, 4), "attributes": {k: round(v, 4) for k, v in attrs.items()}, "feedback": ""}
            f.write(json.dumps(item) + "\n")
    print(f"Created toy dataset at: {out_dir}")


if __name__ == "__main__":
    main()

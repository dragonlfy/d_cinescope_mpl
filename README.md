# CineScope-MPL (Simplified)

A **minimal PyTorch implementation** of the core ideas in **CineScope-MPL: Multimodal Prompt Learning on Vision–Language Foundation Models for Automatic Storyboard Assessment and Formative Feedback in Film Education**.

This repository is intentionally lightweight. It is designed for:
- quickly demonstrating the paper's main method pipeline;
- serving as a clean codebase for anonymous review / supplementary material;
- making it easy to replace the toy encoder with CLIP / BLIP / LLaVA-style backbones later.

> **Note**  
> This is a simplified reference implementation, not a full reproduction of the paper's reported results.  
> The code keeps the key algorithmic structure: **HCPD → COTA → evidence aggregation → score/attribute prediction → template-based formative feedback**.

---

## 1. Method Overview

CineScope-MPL evaluates student storyboards by combining structured cinematographic priors and vision–language prompt learning.

The simplified implementation contains four core modules:

### 1.1 HCPD: Hierarchical Cinematographic Prompt Decomposition

Instead of using a flat sequence of learnable prompts, HCPD organizes cinematographic knowledge as a four-level prompt tree:

```text
shared root context
    ├── composition
    │       ├── rule of thirds
    │       ├── symmetry
    │       └── leading lines
    ├── lighting
    │       ├── key light
    │       ├── contrast
    │       └── shadow
    ├── color
    │       ├── harmony
    │       ├── palette
    │       └── temperature
    ├── motion
    │       ├── subject motion
    │       ├── camera direction
    │       └── pacing
    └── narrative
            ├── intent
            ├── emotion
            └── continuity
```

Each prompt token is synthesized as:

```math
P_{l,k}^{(j)} = \phi_l(c_0 \Vert c_l \Vert e_{l,k}) + \Delta_{l,k}^{(j)}.
```

### 1.2 COTA: Cross-modal Optimal-Transport Alignment

COTA aligns visual patch features with hierarchical prompt tokens by solving an entropy-regularized optimal-transport problem.

The cost matrix is:

```math
C_{ij} = 1 - \cos(f_i^v, g_j^t) + \beta \Omega_{ij}.
```

Then Sinkhorn iterations produce a soft transport plan:

```math
T^* = \mathrm{diag}(a^*) K \mathrm{diag}(b^*), \quad K = \exp(-C / \epsilon).
```

### 1.3 Dimension-wise Evidence Aggregation

The aligned visual evidence is aggregated into five cinematographic dimensions:

```text
composition, lighting, color, motion, narrative
```

These vectors are used for global score regression and attribute prediction.

### 1.4 Feedback Generation

The full paper uses Cinematographic Chain-of-Thought (CCoT) with visual grounding.  
This simplified repository uses a rule-based feedback generator that turns predicted attribute scores into structured text:

```text
Strengths
Areas for improvement
Pedagogical suggestion
```

---

## 2. Repository Structure

```text
cinescope_mpl_simplified/
├── README.md
├── requirements.txt
├── configs/
│   └── cinescope_toy.yaml
├── cinescope/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   └── feedback.py
├── tools/
│   └── make_toy_data.py
├── train.py
├── infer.py
└── examples/
```

---

## 3. Installation

```bash
git clone https://github.com/your-name/CineScope-MPL.git
cd CineScope-MPL

conda create -n cinescope python=3.10 -y
conda activate cinescope

pip install -r requirements.txt
```

---

## 4. Prepare Toy Data

The repository includes a script that creates a tiny synthetic storyboard dataset.

```bash
python tools/make_toy_data.py --out_dir examples/toy_data --num_samples 64
```

It will generate:

```text
examples/toy_data/
├── images/
│   ├── sample_0000.png
│   ├── sample_0001.png
│   └── ...
└── annotations.jsonl
```

Each line in `annotations.jsonl` has the following format:

```json
{
  "image": "images/sample_0000.png",
  "score": 0.73,
  "attributes": {
    "composition": 0.82,
    "lighting": 0.70,
    "color": 0.65,
    "motion": 0.58,
    "narrative": 0.76
  },
  "feedback": "Optional teacher feedback text."
}
```

---

## 5. Train

```bash
python train.py \
  --config configs/cinescope_toy.yaml \
  --data_root examples/toy_data \
  --ann_file examples/toy_data/annotations.jsonl \
  --output_dir checkpoints/toy_run
```

A checkpoint will be saved to:

```text
checkpoints/toy_run/best.pt
```

---

## 6. Inference

```bash
python infer.py \
  --checkpoint checkpoints/toy_run/best.pt \
  --image examples/toy_data/images/sample_0000.png
```

Example output:

```json
{
  "score": 0.72,
  "attributes": {
    "composition": 0.78,
    "lighting": 0.66,
    "color": 0.71,
    "motion": 0.55,
    "narrative": 0.74
  },
  "feedback": {
    "strengths": "...",
    "areas_for_improvement": "...",
    "pedagogical_suggestion": "..."
  }
}
```

---

## 7. Replace Toy Encoder with CLIP

The current `ToyVisualEncoder` is deliberately small. To use a real frozen VLM backbone:

1. Replace `ToyVisualEncoder` in `cinescope/model.py`.
2. Return patch-level visual features with shape:

```python
visual_patches: Tensor  # [B, N_v, D]
```

3. Optionally freeze the visual encoder:

```python
for p in model.visual_encoder.parameters():
    p.requires_grad = False
```

4. Keep HCPD, COTA, and heads unchanged.

---

## 8. Training Objective

The simplified training objective is:

```math
\mathcal{L}
=
\mathcal{L}_{score}
+
\lambda_{attr}\mathcal{L}_{attr}
+
\lambda_{OT}\mathcal{L}_{OT}
+
\lambda_{\Delta}\|\Delta\|_2^2.
```

The full paper additionally includes CCoT feedback loss and a visual grounding loss.

---

## 9. Configuration

Default toy configuration:

```yaml
model:
  image_size: 224
  dim: 128
  patch_size: 16
  num_prompt_tokens: 4
  sinkhorn_iters: 20
  epsilon: 0.05
  beta_prior: 0.1

train:
  epochs: 20
  batch_size: 8
  lr: 0.0005
  weight_decay: 0.01
  lambda_attr: 1.0
  lambda_ot: 0.2
  lambda_delta: 0.0001
```

---

## 10. Citation

```bibtex
@article{cinescope_mpl,
  title={Multimodal Prompt Learning on Vision--Language Foundation Models for Automatic Storyboard Assessment and Formative Feedback in Film Education},
  author={Anonymous},
  journal={Anonymous Submission},
  year={2026}
}
```

---

## 11. License

This simplified implementation is released under the MIT License for academic use.

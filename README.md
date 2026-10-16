# Infraspecific Plant Vision Benchmarking

This repository provides a standardized benchmarking framework for **infraspecific plant classification** (variety and cultivar identification). It evaluates modern Self-Supervised Learning (SSL) architectures (DinoV2) against legacy and modern CNN baselines.

The framework is designed to handle the unique challenges of fine-grained botanical identification, including multi-image observation consensus and organ-specific ablation studies.

---

## 🌟 Key Features

* **Observation-Level Voting**: Aggregates softmax probabilities across multiple images of the same plant individual (using the `tidy` identifier) to improve identification accuracy.
* **Taxonomic Weight Analysis (Organ Ablation)**: Automated support for excluding specific botanical organs (e.g., fruit, leaf, habit) to quantify their contribution to variety identification.
* **Unified Agnostic Pipeline**: Seamlessly switches between datasets (e.g., 87 Banana varieties or 22 Market Garden varieties or 34 Grapevine varieties) using a standardized metadata structure.
* **Modern Benchmarking**: Compares State-of-the-Art (SOTA) Vision Transformers (ViTs) against historical (ResNet50) and modern (ConvNeXt) CNNs.

---

## 🛠️ Installation

We support two ways to set up the environment. **Micromamba** is recommended for research consistency.

### Option A: Micromamba (Recommended)
```bash
# Create and activate the environment from the YAML file
micromamba create -f environment.yml -y
micromamba activate infraspecific-plant-vision
```

### Option B: Pip (Standard)
```bash
# Ensure you have build tools installed on your OS
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
├── data/
│   ├── banana/             # Standardized Zenodo subset of iNaturalist observations (87 varieties)
│   ├── market-garden/      # Novel vegetable variety dataset (22 varieties)
│   └── grape_variety/      # Grapevine variety dataset (34 varieties)
├── models/
│   └── factory.py          # Unified model creator for ViTs and CNNs
├── utils/
│   └── metrics.py          # Observation-level voting and top-k logic
├── scripts/
│   ├── run_benchmarking.sh # Reproduce all architecture comparisons
│   └── run_ablation.sh     # Automate organ-specific analysis
├── train.py                # Training entry point
└── validate.py             # Inference and benchmarking results entry point
├── environment.yml         # Micromamba environment configuration
└── requirements.txt        # Universal Pip dependencies list
```

---

## 🧪 Data & Weights Availability

### Datasets (Zenodo)
Both datasets are hosted on Zenodo to ensure full reproducibility of the train/val/test splits and optimized training performance using `images_mini`.
* **Infraspecific Banana Dataset**: [DOI Link Placeholder] (87 varieties).
* **Market Garden Dataset**: [DOI Link Placeholder] (22 varieties).
* **Grape Variety Dataset**: [DOI Link Placeholder] (34 varieties).

### Model Weights
| Architecture | Pre-training | Accessibility |
| :--- | :--- | :--- |
| **DinoV2 (Base/Reg4)** | SSL (General) | **Public** (via `timm`) |
| **ResNet50 / ConvNeXt** | Supervised | **Public** (via `timm`) |
| **PlantCLEF 2024** | Botanical SSL | **Public** (Zenodo) |
| **Pl@ntNet-SSL/Sup** | Botanical SSL/Sup | **Private** (Internal Only) |

---

## 🚀 Usage

### 1. Running Benchmarks
The provided shell scripts are dataset-agnostic. To switch between datasets, edit the `DATASET` variable in `scripts/run_benchmarking.sh`.
```bash
# Reproduce all experiments for the selected dataset
bash scripts/run_benchmarking.sh
```

### 2. Manual Training (Fine-tuning)
```bash
python train.py --model vit_base_patch14_dinov2.lvd142m \
                --root-dir ./data/market-garden/images_mini/train \
                --metadata-train ./data/market-garden/train.csv \
                --frozen
```

### 3. Organ Ablation Study
Measure the impact of removing a specific organ (e.g., 'fruit') from the training data:
```bash
python train.py --exclude-organ fruit [additional arguments...]
```

---

## 📝 Citation

If you use this code or our datasets in your research, please cite our work as follows:

```bibtex
@article{tibi2026geometric,
title = {On the role of geometric representations for ultra-fine-grained crop variety identification},
journal = {Smart Agricultural Technology},
volume = {15},
pages = {102442},
year = {2026},
issn = {2772-3755},
doi = {https://doi.org/10.1016/j.atech.2026.102442},
url = {https://www.sciencedirect.com/science/article/pii/S2772375526006672},
author = {Charlotte Tibi and Ilyass Moummad and Alexis Joly and Yves Caraglio and Christophe Jenny and Pierre Bonnet and Hervé Goëau},
}```

---
## ⚖️ License
This project is licensed under the MIT License - see the `LICENSE` file for details.


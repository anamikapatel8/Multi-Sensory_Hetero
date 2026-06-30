# Data- and Parameter-Efficient Multisensory Learning with Heterogeneous Fusion of SAR and Multi-spectral Foundation Models for Cloud-Resilient Earth Observation

**Authors:** Anamika Patel, Sudipan Saha
Department of Electrical Engineering & School of Artificial Intelligence, IIT Delhi

---

## Overview

This repository contains the implementation of a parameter-efficient, heterogeneous multi-foundation pipeline that pairs a frozen cross-modal model ([CROMA](https://arxiv.org/abs/2311.00566)) with a frozen SAR-specialized model ([SARATR-X](https://arxiv.org/abs/2405.09365)) for cloud-resilient Earth Observation. We introduce a **Context-Aware Gated Fusion (CAGF)** mechanism that dynamically weights radar and multi-spectral token streams based on a per-chip cloud-cover ratio, and evaluate the framework across three downstream tasks:

1. Binary flood segmentation (C2SMS-Floods)
2. Multi-temporal crop-type panoptic segmentation (PASTIS-R)
3. Maritime oil spill classification (Sentinel-1 SAR)

All backbone parameters remain frozen across every experiment; only lightweight task heads, adapters, and the fusion module are trained (≤16.56M trainable parameters per task).

---

## Repository Contents

| Notebook | Description | Paper Section |
|---|---|---|
| `1a_Flood_detection_c2sms_CROMA_dualbranch.ipynb` | Core CAGF architecture: frozen CROMA + SARATR-X dual-encoder fusion conditioned on cloud-cover ratio. Defines the gating mechanism, spatial alignment between the two backbones, and the shared fusion module used downstream. | Sec. IV.B–D (Methodology: Spatial Alignment, CAGF) |
| `1b_Flood_detection_c2sms_CROMA_SARATR-X.ipynb` | Flood segmentation pipeline on C2SMS-Floods. Implements cloud-aware target selection, the composite CE+Dice loss, and the segmentation head producing 512×512 flood maps. Reproduces Table I and Figs. 3–4. | Sec. IV.E, V.A (Flood Detection) |
| `2_Oil_Spill_Detection.ipynb` | Maritime oil spill classification using frozen SARATR-X as a feature extractor with a lightweight MLP head. Includes Optuna hyperparameter search and weighted cross-entropy for class imbalance. Reproduces Table II and Fig. 5. | Sec. IV.G, V.B (Oil Spill Detection) |
| `3_Crop-type_Panoptic_Segmentation_PASTIS-R.ipynb` | Multi-temporal crop-type panoptic segmentation on PASTIS-R. Implements the Temporal Attention module, bidirectional cross-attention fusion, and the convolutional upsampling segmentation head over 12 growing-season dates. Reproduces Table III and Fig. 6. | Sec. IV.F, V.C (PASTIS-R) |
---

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| C2SMS-Floods | [Registry of Open Data on AWS](https://registry.opendata.aws/c2smsfloods/) | Co-registered Sentinel-1/Sentinel-2 triplets with per-chip cloud masks and binary flood labels. |
| PASTIS-R | [Zenodo](https://zenodo.org/) (see [paper repo](https://github.com/VSainteuf/pastis-benchmark)) | Multi-temporal Sentinel-1/Sentinel-2 time series with 20-class panoptic crop labels over French agricultural parcels. |
| Sentinel-1 SAR Oil Spill Dataset | [Kaggle](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset) | Binary Oil Spill / No Spill chips from maritime regions. |

Datasets are not redistributed in this repository due to size and licensing; please download from the original sources linked above and update the data paths at the top of each notebook.

---

## Pretrained Foundation Models

| Model | Source | Usage |
|---|---|---|
| CROMA (base) | [github.com/antofuller/CROMA](https://github.com/antofuller/CROMA) | Frozen multi-spectral / cross-modal encoder, patch size 8, embedding dim 768. |
| SARATR-X | [huggingface.co/waterdisappear/SARATR-X](https://huggingface.co/waterdisappear/SARATR-X) | Frozen SAR-specialized HiViT encoder, reinitialized at patch size 8 for token-grid alignment with CROMA. |

Pretrained checkpoints must be downloaded separately following the instructions in the respective official repositories.

---

## Environment

All notebooks were developed and run on Kaggle/Colab-style environments with a single GPU (≥16GB VRAM recommended for PASTIS-R training). Key dependencies:

```
torch >= 2.0
torchvision
numpy
einops
optuna          # for oil spill hyperparameter search
scikit-learn
matplotlib
```

Exact package versions are listed at the top of each notebook where relevant.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{patel2026heterogeneous,
  title     = {Data- and Parameter-Efficient Multisensory Learning with Heterogeneous
               Fusion of SAR and Multi-spectral Foundation Models for
               Cloud-Resilient Earth Observation},
  author    = {Patel, Anamika and Saha, Sudipan},
  booktitle = {Proceedings of the IEEE India Geoscience and Remote Sensing
               Symposium (InGARSS)},
  year      = {2026}
}
```

---

## Acknowledgements

We thank the creators of CROMA and SARATR-X for releasing their pretrained checkpoints, and the teams behind C2SMS-Floods, PASTIS-R, and the Kaggle Sentinel-1 SAR oil spill dataset for making their benchmarks publicly available. We also thank the IIT Delhi HPC facility for compute resources.

## Contact

For questions, please contact Anamika Patel at anamikapatel8@gmail.com.

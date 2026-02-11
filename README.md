# LiteAVSEMamba

**Lightweight Audio-Visual Speech Enhancement with Mamba**

University of Bristol | Feb 2026

**Team:** Billy (Lead), Dominic, Ronny, Fan, Shunjie, Zhenning

> **Branch `SEMamba`** — Development base built on [SEMamba](https://github.com/RoyChao19477/SEMamba) (Chao et al., IEEE SLT 2024). All new module files contain TODO skeletons only. See each file's header for your assigned tasks, design specs, and paper references.

---

## Overview

LiteAVSEMamba is a lightweight audio-visual speech enhancement system built on top of [SEMamba](https://arxiv.org/abs/2405.06573) [1]. We extend the audio-only Mamba architecture with visual information from face video, using two novel fusion modules: **VCE** (Visual Confidence Estimator) and **FSVG** (Frequency-Selective Visual Gating).

### Core Innovation: Double-Gated Fusion

```
fused = audio_feat + alpha * gate * visual_feat
```

- `alpha` (VCE): per-frame visual reliability score — when visual input is degraded, alpha -> 0, system falls back to audio-only
- `gate` (FSVG): per-frequency gating weight — speech-relevant frequencies (300Hz-3kHz) receive stronger visual influence

### Design Targets

| Property | Value | Source |
|----------|-------|--------|
| SEMamba baseline params | ~10.5M | [SEMamba repo](https://github.com/RoyChao19477/SEMamba) |
| Complexity | O(N) linear | Mamba [2] proven property |
| Visual Input | 96x96 RGB full-face @ 25fps | Config |
| Audio Input | 16kHz mono | Config |
| Streaming | Design goal (time-causal Mamba) | TBD after implementation |
| Parameter reduction | Design goal (significant reduction from SEMamba) | TBD after implementation |

---

## Architecture

```
 noisy_audio [B, 16000]              face_video [B, 3, 25, 96, 96]
      |                                      |
      v                                      v
 +-----------+                    +-------------------------+
 |   STFT    |  n_fft=400        | LiteVisualEncoder [NEW] |  Custom 3D CNN
 +-----------+  hop=100          | 4-layer Conv3d          |  Ref: [6]
      |                           +-------------------------+
 mag + pha                              |              |
 [B, 2, T, F]                     visual_raw       visual_feat
      |                           [B, 512, T]     [B, 64, T, F]
      v                                |
 +------------------+                  v
 |  DenseEncoder    |        +-------------------+
 +------------------+        | VCE [NEW]         |  Visual Confidence
      |                       | MLP + causal Conv1d|  Ref: [13][14]
 [B, 64, T, F//2]            +-------------------+
 = audio_feat                         |
      |                          alpha [B, T, 1]
      |                               |
      |    +--------------+           |
      +--->| FSVG [NEW]   |           |
      |    | Conv2d gate   |           |
      |    | + freq prior  |           |
      |    +--------------+           |
      |          |                    |
      |     gate [B,1,T,F]           |
      v          v                    v
 +--------------------------------------------------+
 | DOUBLE-GATED FUSION [NEW]                         |
 |  fused = audio_feat + alpha * gate * visual_feat  |
 |  Ref: FiLM [13], Gated Multimodal Fusion [14]    |
 +--------------------------------------------------+
      |
 4x CausalTFMambaBlock [MODIFIED]
 (Time: unidirectional / Freq: bidirectional)
      |
 MagDecoder + PhaseDecoder -> iSTFT -> enhanced_audio
```

---

## Complete File Map

Legend: `[BASE]` = SEMamba original, `[NEW]` = our addition, `[EXT]` = extended from original

```
LiteAVSEMamba/
│
├── README.md                                  [NEW]  Project overview + architecture + file map
├── .gitignore                                 [EXT]  Ignore rules (added exp/, data/, *.pth, etc.)
├── LICENSE                                    [BASE] MIT license from SEMamba
├── requirements.txt                           [BASE] Python dependencies (torch, mamba-ssm, pesq, etc.)
│
├── models/
│   ├── generator.py                           [BASE] SEMamba class (audio-only SE model)
│   │
│   ├── mamba_block.py                         [BASE] MambaBlock (bidirectional SSM scan)
│   │                                                 TFMambaBlock (time-bi + freq-bi Mamba)
│   │
│   ├── vce.py                                 [NEW]  Visual Confidence Estimator (Billy)
│   │                                                 (per-frame alpha: "is this frame's video reliable?")
│   │
│   ├── fsvg.py                                [NEW]  Freq-Selective Visual Gating (Dominic)
│   │                                                 (per-frequency gate: "does this freq need visual?")
│   │
│   ├── lite_visual_encoder.py                 [NEW]  Lightweight 3D CNN visual encoder (Ronny)
│   │                                                 (face video [B,3,T,96,96] -> visual features)
│   │
│   ├── loss.py                                [BASE] phase_losses, pesq_score (spectral loss + eval metric)
│   │
│   ├── codec_module.py                        [BASE] DenseEncoder (audio feat extractor, Conv2d+DenseBlock)
│   │                                                 MagDecoder (magnitude mask predictor)
│   │                                                 PhaseDecoder (phase estimator via atan2)
│   │
│   ├── discriminator.py                       [BASE] MetricDiscriminator (PESQ-guided adversarial loss,
│   │                                                 used in SEMamba training only, not in our model)
│   │
│   ├── stfts.py                               [BASE] STFT/iSTFT with power compression
│   │                                                 (audio waveform <-> mag+phase spectrogram)
│   │
│   ├── lsigmoid.py                            [BASE] LearnableSigmoid (sigmoid with learnable slope,
│   │                                                 used in MagDecoder mask output, from MP-SENet)
│   │
│   └── pcs400.py                              [BASE] Perceptual Contrast Stretching (per-freq-bin
│                                                     weighting table, SEMamba preprocessing variant,
│                                                     not used in our model)
│
├── dataloaders/
│   ├── dataloader_vctk.py                     [BASE] VCTK-DEMAND audio-only dataloader
│   └── dataloader_av.py                       [NEW]  Audio-Visual dataloader (Fan)
│                                                     (paired audio+video, on-the-fly noise mixing,
│                                                     visual augmentation for VCE training)
│
├── recipes/
│   ├── SEMamba_advanced/                       [BASE] SEMamba audio-only configs
│   │   ├── SEMamba_advanced.yaml                     (main config: STFT, model, training params)
│   │   └── SEMamba_advanced_pretrainedD.yaml         (config with pretrained discriminator)
│   ├── SEMamba_advanced_PCS/                   [BASE] SEMamba + PCS preprocessing config
│   │   └── SEMamba_advanced_PCS.yaml                 (enables use_PCS400, not used by us)
│   └── LiteAVSE/                              [NEW]  Our AV model configs
│       └── LiteAVSE.yaml                             (model arch + training + data settings)
│
├── train.py                                   [BASE] SEMamba training loop (audio-only, with GAN loss)
├── train_lite.py                              [NEW]  LiteAVSE training loop (Collaborative)
│                                                     (AV training with 6-component loss)
├── inference.py                               [BASE] SEMamba inference script (load ckpt -> enhance audio)
│
├── utils/
│   └── util.py                                [BASE] Config loader, seed init, GPU info, distributed setup
│
├── run.sh                                     [BASE] SEMamba training launch script
├── runPCS.sh                                  [BASE] SEMamba+PCS training launch script
├── make_dataset.sh                            [BASE] Generate JSON data lists from VCTK-DEMAND
└── pretrained.sh                              [BASE] Download SEMamba pretrained checkpoints
```

### What We Changed (Summary)

| Type | Count | Files |
|------|-------|-------|
| **[NEW] files** | 6 | `vce.py`, `fsvg.py`, `lite_visual_encoder.py`, `dataloader_av.py`, `train_lite.py`, `LiteAVSE.yaml` |
| **[EXT] extended** | 1 | `.gitignore` (added exp/, data/, *.pth, etc.) |
| **[BASE] unchanged** | 19 | All original SEMamba files |

---

## Quick Start

### 1. Environment Setup

```bash
git clone -b SEMamba https://github.com/BillyHou1/LiteAVSE.git
cd LiteAVSE
pip install -r requirements.txt

# Install Mamba SSM (see mamba_install/ for instructions)
```

### 2. Datasets

**Speech (Audio-Visual):**

| Dataset | Size | Purpose | Access |
|---------|------|---------|--------|
| [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/) | 28h, single-speaker | Prototype / ablation | Public |
| [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) | 224h, multi-speaker | Full training | Requires application |
| [LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) | 438h, multi-speaker | Full training | **Currently unavailable** (Oxford VGG access suspended) |

**Noise:**

| Dataset | Size | Notes |
|---------|------|-------|
| [DEMAND](https://zenodo.org/record/1227121) | 18 types | Limited diversity; sufficient for prototyping |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) | 65,000+ clips | Large-scale; recommended for full training |
| [MUSAN](https://www.openslr.org/17/) | Music, speech, noise | Complements DEMAND |

> **Note:** DEMAND alone may not provide sufficient noise diversity for robust evaluation. Consider combining with DNS Challenge or MUSAN for full-scale experiments. LRS3 access status should be confirmed with Oxford VGG before planning experiments that depend on it.

Prepare JSON lists per dataset: `data/<dataset>_train.json`, `data/<dataset>_valid.json`

### 3. Training

```bash
# Audio-only baseline (SEMamba — works out of the box)
python train.py --config recipes/SEMamba_advanced/SEMamba_advanced.yaml

# Audio-visual (after implementing TODO modules)
python train_lite.py --config recipes/LiteAVSE/LiteAVSE.yaml \
    --exp_folder exp --exp_name LiteAVSE_v1
```

---

## Module Implementation Guide

Each new module file contains TODO comments with:
- **Purpose** — what the module does
- **References** — which paper sections to read before implementing
- **Integration** — how it connects to other modules

**Recommended reading order:**
1. SEMamba paper [1] — understand the base architecture
2. Mamba paper [2] — understand SSM mechanism
3. Your assigned module's references (listed in file header)

---

## References

[1] R. Chao et al., "An Investigation of Incorporating Mamba for Speech Enhancement," IEEE SLT, 2024.

[2] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," COLM, 2024.

[3] T. A. Ma et al., "Real-Time Audio-Visual Speech Enhancement Using Pre-trained Visual Representations," Interspeech, 2025.

[4] Y.-X. Lu et al., "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra," Interspeech, 2023.

[5] K. Li et al., "An Audio-Visual Speech Separation Model Inspired by Cortico-Thalamo-Cortical Circuits," IEEE TPAMI, 2024.

[6] P. Ma et al., "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels," IEEE ICASSP, 2023.

[7] A. Howard et al., "Searching for MobileNetV3," IEEE/CVF ICCV, 2019.

[8] G. Huang et al., "Densely Connected Convolutional Networks," IEEE CVPR, 2017.

[9] J. Le Roux et al., "SDR -- Half-baked or Well Done?" IEEE ICASSP, 2019.

[10] B. Shi et al., "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction," ICLR, 2022.

[11] C. H. Taal et al., "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech," IEEE TASLP, 2011.

[12] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," Interspeech, 2019.

[13] E. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI, 2018.

[14] J. Arevalo et al., "Gated Multimodal Units for Information Fusion," ICLR Workshop, 2017.

[15] A. W. Rix et al., "Perceptual Evaluation of Speech Quality (PESQ)," IEEE ICASSP, 2001.

---

## License

Based on [SEMamba](https://github.com/RoyChao19477/SEMamba) by R. Chao et al.

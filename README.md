# LiteAVSEMamba

**Lightweight Audio-Visual Speech Enhancement with Mamba**

University of Bristol  
Feb 2026

**Team:** Billy, Dominic, Ronny, Fan, Shunjie, Zhenning

> **Branch `SEMamba`** — Development base built on [SEMamba](https://github.com/RoyChao19477/SEMamba) (Chao et al., IEEE SLT 2024). All new module files contain TODO skeletons only. See each file's header for your assigned tasks, design specs, and paper references.

---

## Overview

LiteAVSEMamba is a lightweight audio-visual speech enhancement system that builds on SEMamba [1]. We take the original audio-only Mamba architecture and bring in visual information from face video, using two new fusion modules we designed: VCE (Visual Confidence Estimator) and FSVG (Frequency-Selective Visual Gating).

### Core Innovation: Double-Gated Fusion

```
fused = audio_feat + alpha * gate * visual_feat
```

- `alpha`(VCE): a score for each frame that tells the system how trustworthy the visual input is. If the video is poor or the face is obscured, alpha drops to zero and the system just uses audio instead.
- `gate`(FSVG): a weight for each frequency that controls how much visual information gets mixed in. Frequencies where speech lives (300 Hz–3 kHz) get a bigger boost from the visual side.
### Design Targets

|Property|Value|Source|
|----------|-------|--------|
|SEMamba baseline params|~10.5M| [SEMamba repo](https://github.com/RoyChao19477/SEMamba) |
|Complexity|O(N) linear|Mamba[2]proven property|
|Visual Input|96x96 RGB full-face @ 25fps|Config|
|Audio Input|16kHz mono|Config|
|Streaming|Design goal (time-causal Mamba)|TBD|
|Parameter reduction|Design goal (significant reduction from SEMamba)|TBD|

---

## Expect Architecture


<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/349352a5-6164-4377-becf-81902935bff9" />


---

## Draft Complete File Map

Legend:`[BASE]`= SEMamba original, `[NEW]`= our addition, `[EXT]`= extended from original

```
LiteAVSEMamba/
│
├── README.md                                  [NEW]  Project overview + architecture + file map
├── .gitignore                                 [EXT]  Ignore rules
├── LICENSE                                    [BASE] MIT license from SEMamba
├── requirements.txt                           [BASE] Python dependencies
│
├── models/
│   ├── generator.py                           [EXT]  SEMamba class [BASE] (audio-only)
│   │                                                 + LiteAVSEMamba class [NEW]
│   │                                                   (audio-visual model with double-gated fusion)
│   │
│   ├── mamba_block.py                         [EXT] MambaBlock [BASE] (bidirectional SSM scan)
│   │                                                 TFMambaBlock [BASE] (time-bi+freq-bi Mamba)
│   │                                                 + CausalTFMambaBlock[NEW]
│   │                                                   (time-UNI+freq-bi, enables streaming)
│   │
│   ├── vce.py                                 [NEW]  Visual Confidence Estimator
│   │                                                 (per-frame alpha: is this frame's video reliable?)
│   │
│   ├── fsvg.py                                [NEW]  Freq-Selective Visual Gating
│   │                                                 (per-frequency gate: does this freq need visual?)
│   │
│   ├── lite_visual_encoder.py                 [NEW]  Lightweight 3D CNN visual encoder
│   │                                                 (face video[B,3,T,96,96] -> visual features)
│   │
│   ├── loss.py                                [EXT]  phase_losses, pesq_score [BASE]
│   │                                                 + si_sdr_loss, stoi_score [NEW]
│   │
│   ├── codec_module.py                        [BASE] DenseEncoder(audio feat extractor, Conv2d+DenseBlock)
│   │                                                 MagDecoder(magnitude mask predictor)
│   │                                                 PhaseDecoder(phase estimator via atan2)
│   │
│   ├── discriminator.py                        [BASE] MetricDiscriminator(PESQ-guided adversarial loss,
│   │                                                 used in SEMamba training only, not in ours)
│   │
│   ├── stfts.py                               [BASE] STFT,iSTFT with power compression
│   │                                                 (audio waveform <-> mag+phase spectrogram)
│   │
│   ├── lsigmoid.py                            [BASE] LearnableSigmoid(sigmoid with learnable slope,
│   │                                                 used in MagDecoder mask output, from MP-SENet)
│   │
│   └── pcs400.py                              [BASE] Perceptual Contrast Stretching(per-freq-bin
│                                                     weighting table, SEMamba preprocessing variant,
│                                                     not used in ours)
│
├── dataloaders/
│   ├── dataloader_vctk.py                     [BASE] VCTK-DEMAND audio-only dataloader
│   └── dataloader_av.py                       [NEW]  Audio-Visual dataloader
│                                                     (paired audio and video, noise is mixed in on the fly,
│                                                     visual augmentations are applied to help train the VCE module)
├── recipes/
│   ├── SEMamba_advanced/                       [BASE] SEMamba audio-only configs
│   │   ├── SEMamba_advanced.yaml                    (main config STFT, model, training params)
│   │   └── SEMamba_advanced_pretrainedD.yaml         (config with pretrained discriminator)
│   ├── SEMamba_advanced_PCS/                   [BASE] SEMamba+PCS preprocessing config
│   │   └── SEMamba_advanced_PCS.yaml                 (enables use_PCS400)
│   └── LiteAVSE/                              [NEW]  Our AV model configs
│       └── LiteAVSE.yaml                             (model arch + training + data + eval settings)
│
├── train.py                                   [BASE] SEMamba training loop (audio-only, with GAN loss)
├── train_lite.py                              [NEW]  LiteAVSE training loop
│                                                     (AV training, component loss, PESQ/STOI/SI-SDR eval)
├── inference.py                               [BASE] SEMamba inference script (load ckpt -> enhance audio)
│
├── utils/
│   └── util.py                                [BASE] Config loader, seed init, GPU info, distributed setup
│
├── run.sh                                     [BASE] SEMamba training launch script
├── runPCS.sh                                  [BASE] SEMamba+PCS training launch script
├── make_dataset.sh                            [BASE] Generate JSON data lists from VCTK-DEMAND
└── pretrained.sh                              [BASE] Download SEMamba pretrained checkpoint
```

### What We will change

| Type | Count(Expect) | Files |
|------|-------|-------|
|[NEW] files|6|`vce.py`, `fsvg.py`, `lite_visual_encoder.py`, `dataloader_av.py`, `train_lite.py`, `LiteAVSE.yaml`|
|[EXT] extended|3|`generator.py` +LiteAVSEMamba, `mamba_block.py` +CausalTFMamba, `loss.py` +SI-SDR/STOI|
|[BASE] unchanged|17|All original SEMamba files|

---

## Quick Start
### 1. Environment Setup
```bash
git clone -b SEMamba https://github.com/BillyHou1/LiteAVSE.git
cd LiteAVSE
pip install -r requirements.txt
# Install Mamba SSM
```
### 2. Datasets
**Speech:**
| Dataset | Size | Purpose | Access |  
|---------|------|---------|--------|  
|[GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/)|28h,single-speaker|Prototype & ablation|Public|  
|[LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)|224h, multi-speaker|Full training|Requires application|  
|[LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)|438h, multi-speaker|Full training|Currently unavailable|  

**Noise:**  
| Dataset | Size | Notes |  
|---------|------|-------|  
|[DEMAND](https://zenodo.org/record/1227121) |18 types|Limited diversity|  
|[DNS Challenge](https://github.com/microsoft/DNS-Challenge)|65000+ clips|Large-scale, for full training|  
|[MUSAN](https://www.openslr.org/17/)|Music, speech, noise| Complements DEMAND|  
> **Note:** DEMAND on its own doesn't cover enough types of noise to really stress-test the system. For proper experiments, it's worth mixing in DNS Challenge or MUSAN as well. Also, LRS3 not available.

     Prepare JSON lists per dataset: `data/<dataset>_train.json`, `data/<dataset>_valid.json`

### 3.Training

```bash
# Audio-only baseline (SEMamba)
python train.py --config recipes/SEMamba_advanced/SEMamba_advanced.yaml

# Audio-visual (TO-DO)
python train_lite.py --config recipes/LiteAVSE/LiteAVSE.yaml \
    --exp_folder exp --exp_name LiteAVSE_v1
```
---
## Module Implementation Guide
Each new module file contains TODO comments with:
- **Purpose**- what the module does
- **References**- which paper sections to read before implementing
- **Integration**- how it connects to other modules

**Recommended reading order:**
1. SEMamba paper[1]: understand the base architecture
2. Mamba paper[2]: understand SSM mechanism
3. Your assigned module's references, these listed in file header
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
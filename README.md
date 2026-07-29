# EdgeFace-Multimodal: Quantized Low-Rank Tensor Fusion (Q-LRTF)

This repository contains the official implementation of **EdgeFace-Multimodal**, extending the baseline EdgeFace architecture with a novel **Quantized Low-Rank Tensor Fusion (Q-LRTF)** mechanism for highly efficient, dual-stream (Visible + Near-Infrared) face recognition. 

This work targets the rigorous standards of the *Information Fusion* journal, offering state-of-the-art multimodal fusion at a fraction of the computational complexity of standard bilinear pooling or concatenation methods.

---

## 🚀 Key Highlights

* **Multimodal Tensor Fusion**: Introduced a low-rank tensor product mechanism $O(r(2d+h))$ that combines dual-stream VIS and NIR modalities efficiently without geometric parameter explosions.
* **Cross-Modal Scale Bounding (CMSB) Loss**: Integrated an auxiliary regularization penalty to ensure symmetric gradient flow and prevent modality collapse during dual-stream training.
* **Simulated Quantization (Fake INT8)**: Integrated native PyTorch dynamic quantization into the low-rank tensor fusion operation, preserving mathematical correctness and avoiding compounding quantization error during edge device inference.
* **Universal TIMM Mobile Backbones**: Expanded the dynamic configuration parser to enable arbitrary TIMM CNN and Transformer models (e.g. MobileNetV3, ConvNeXt) as multimodal feature extractors.

---

## 📊 Performance & Compression Summary (LFW Benchmark)

The following table summarizes the trade-offs between computational complexity (MFLOPs), model parameters, disk footprint, and verification accuracy on the Labeled Faces in the Wild (LFW) dataset:

| Model Configuration | Description | Complexity (MFLOPs) | Parameter Count (M) | Disk Footprint (MB) | FLOPs Reduction | LFW Verification Accuracy | Accuracy Loss vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`edgeface_xs_gamma_06`** | XS Baseline | 153.998 | 1.770 | 7.18 | Baseline | 99.733% +- 0.35% | - |
| **`edgeface_xs_gamma_06_q`** | **XS Low-Rank + Dynamic INT8** | **20.160** | **0.202** | **2.58** | **7.6x Lower** | **99.617% +- 0.34%** | **0.116%** |
| **`edgeface_fusion_xs_gamma_06`** | Dual-Stream Q-LRTF (Float) | 616.38 | 3.640 | 14.60 | Baseline | - | - |
| **`edgeface_fusion_xs_gamma_06_q`** | **Dual-Stream Q-LRTF (INT8)** | **81.12** | **0.503** | **2.21** | **7.6x Lower** | - | - |
| **`edgeface_s_gamma_05`** | S Baseline | 306.116 | 3.653 | 14.71 | Baseline | 99.783% +- 0.27% | - |
| **`edgeface_s_gamma_05_q`** | **S Low-Rank + Dynamic INT8** | **36.284** | **0.419** | **5.13** | **8.4x Lower** | **99.500% +- 0.39%** | **0.283%** |

*Note: Quantized parameter counts reflect only unquantized parameters (like batch-normalization variables and convolution filters), as dynamic INT8 weights are packed into scale/offset integer buffers.*

---

## ⚙️ Installation & Environment Setup

This codebase is configured to run stably on high-performance compute nodes (PBS queue managers) using CUDA 12.0+ host drivers and modern GPUs:

```bash
# 1. Create and activate your conda environment
conda create -n GPU-env python=3.10 -y
conda activate GPU-env

# 2. Install PyTorch compiled with CUDA 11.8 (provides backward compatibility for sm_90/Hopper)
pip install torch==2.4.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Install core requirements
pip install timm==0.6.12 pandas tabulate mxnet fpdf2 scikit-learn scipy
```

---

## 🔍 Model Evaluation & Profiling

To evaluate model parameter counts, CPU FLOP complexity, and disk storage sizing, run:

```bash
# Profile baseline models
python eval_edgeface.py edgeface_xs_gamma_06
python eval_edgeface.py edgeface_s_gamma_05

# Profile quantized hybrid models
python eval_edgeface.py edgeface_xs_gamma_06_q
python eval_edgeface.py edgeface_s_gamma_05_q
```

---

## 📈 Face Verification Accuracy Testing

We evaluate verification accuracy on standard InsightFace evaluation datasets (binary serialization). 

1. Place your dataset binary (e.g., `lfw.bin`) under the `data/` directory.
2. Run your verification testing pipeline:

```bash
# Run verification accuracy validation
python eval/verification.py --data-dir data/ --target lfw --batch-size 32
```

---

## 🏋️ Training Pipelines

### 1. Synthetic Validation Run (Hopper / GPU Check)
Use the synthetic configuration to test DDP training, VRAM allocation, and logging endpoints without needing a raw face dataset:

```bash
python -m torch.distributed.run --nproc_per_node=1 train_v2.py configs/baseline_synthetic.py
```

### 2. Full Dataset Training (WebFace12M / WebFace4M)
Download the `.rec` files into `data/webface12m/` or `data/webface4m/` and run:

```bash
# Train EdgeFace-S
torchrun --nproc_per_node=8 train_v2.py configs/edgeface_s_gamma_05.py

# Restart from intermediate checkpoint
torchrun --nproc_per_node=8 train_v2_restart.py configs/edgeface_s_gamma_05_restart.py
```

---

## 📂 Core Repository Structure

* `backbones/`: Contains backbone neural architectures, MobileFaceNet wrappers, and low-rank layer factorizations (`timmfr.py`).
* `configs/`: Contains training hyperparameters, learning schedules, and dataset links.
* `eval/`: Exposes evaluation metrics, ROC loaders, and verification metrics.
* `dataset.py`: Decodes RecordIO files and implements synthetic data loading fallbacks.
* `losses.py`: Implements margin-based losses (ArcFace, CosFace).
* `train_v2.py`: Main DDP training pipeline.
* `nas_gamma_search.py`: evolutionary layer-wise gamma rank-ratio optimizer.

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

| Model Configuration | Description | Complexity (MFLOPs) | Parameter Count (M) | Disk Footprint (MB) | Disk Reduction | LFW Verification Accuracy | Accuracy Loss vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`edgeface_xs_gamma_06`** | XS Baseline | 154.00 | 1.770 | 7.18 | Baseline | 99.733% +- 0.35% | - |
| **`edgeface_xs_gamma_06_q`** | **XS Low-Rank + Dynamic INT8** | **154.00** | **1.770** | **2.57** | **2.8x Lower** | **99.617% +- 0.34%** | **0.116%** |
| **`edgeface_fusion_xs_gamma_06`** | Dual-Stream Q-LRTF (Float) | 308.19 | 3.640 | 14.76 | Baseline | - | - |
| **`edgeface_fusion_xs_gamma_06_q`** | **Dual-Stream Q-LRTF (INT8)** | **308.39** | **3.640** | **5.55** | **2.7x Lower** | - | - |
| **`edgeface_s_gamma_05`** | S Baseline | 306.12 | 3.653 | 14.71 | Baseline | 99.783% +- 0.27% | - |
| **`edgeface_s_gamma_05_q`** | **S Low-Rank + Dynamic INT8** | **306.12** | **3.653** | **5.13** | **2.9x Lower** | **99.500% +- 0.39%** | **0.283%** |

*Note: FLOPs are measured for a single 1x3x112x112 input (dual-stream models run both streams on it). Dynamic INT8 keeps the operation count unchanged; it only stores `nn.Linear` weights in 8 bits, so the gain is in disk/memory footprint, not FLOPs or parameter count. Packed INT8 linear weights are included in the parameter count, and their FLOPs are counted with forward hooks because the PyTorch profiler does not report FLOPs for `quantized::linear_dynamic`.*

---

## 🌡️ Thermal-to-Visible HFR Study (paper draft in `paper/`)

A reproducible study of lightweight thermal-to-visible heterogeneous face recognition with EdgeFace-XS, built only on the openly licensed (CC BY 4.0) [SpeakingFaces](https://huggingface.co/datasets/issai/Speaking_Faces) corpus: subject-disjoint 5-fold cross-validation, 3 seeds, 8 adaptation strategies, visible-light forgetting on LFW, and full / stage-selective dynamic INT8.

| Method | Trainable params | Rank-1 FP32 | TAR@FAR=0.1% FP32 | Rank-1 INT8 | Rank-1 INT8 (last stage only) | LFW |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| Zero-shot | 0 | 34.0 | 6.3 | 28.8 | 33.1 | 99.73 |
| Full fine-tuning | 1.77M | 95.6 | 83.8 | 93.4 | 95.6 | 98.52 |
| LoRA (rank +1) | 20.5k | 93.6 | 77.8 | 89.1 | 93.1 | 98.96 |
| Modality-specific LayerNorm | 5.1k | 92.1 | 66.1 | 84.7 | 90.9 | 99.73 |
| Rank-space modulation + msLN | 10.4k | 91.3 | 65.4 | 86.2 | 90.3 | 99.73 |

Full table: `results/hfr/cv_summary.md`. Reproduction steps and the source of every number in the paper: `paper/README.md`.

* `scripts/hfr/download_speakingfaces.py`, `align_speakingfaces.py`: fetch only the needed frames (~3 GB) and align thermal/visible pairs with one MTCNN warp.
* `scripts/hfr/train_hfr.py`, `eval_hfr.py`, `eval_vis_bench.py`: adaptation, HFR protocol, LFW forgetting.
* `scripts/hfr/run_experiments.py` + `configs/*.txt`: resumable experiment runner; `aggregate.py`: mean ± std tables; `benchmark_latency.py`: CPU latency and model size.
* `backbones/modality.py` (rank-space modulation, modality-specific LayerNorm), `backbones/peft.py` (LoRA for factorized layers), `backbones/qat.py` (exact simulation of PyTorch dynamic INT8).

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

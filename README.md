# MobileNet-v2 Compression on CIFAR-10 (CS6886W Assignment 2)

Custom post-training quantization and sub-byte bit-packing pipeline implemented in PyTorch (without external compression APIs) for a CIFAR-10 adapted **MobileNet-v2** architecture.

* **GitHub Repository:** [https://github.com/cs25m504/mobilenetv2](https://github.com/cs25m504/mobilenetv2)

---

## 1. Repository Structure

```text
mobilenetv2/
├── model.py              # CIFAR-10 adapted MobileNet-v2 (modified early strides for 32x32 input)
├── dataset.py            # CIFAR-10 dataloaders, augmentation pipeline, and normalization
├── train.py              # Baseline FP32 training script with Cosine Annealing & checkpointing
├── compress.py           # Custom per-channel weight & dynamic activation quantization + bit-packing
├── memory_profiler.py    # Exact in-memory RAM/VRAM profiler (weights, metadata, and activations)
├── test.py               # CLI evaluation script for compressed models + Weights & Biases logging
├── utils.py              # Deterministic seeding, Top-1 accuracy evaluation, and curve plotting
├── requirements.txt      # Pinned Python dependencies
└── README.md             # Project documentation and reproduction instructions
```

---

## 2. Environment Setup & Dependencies

### Hardware & Software Environment
* **Platform:** Kaggle GPU (`CUDA`) / Linux x86_64
* **Python:** `3.10+`
* **Core Libraries:** `torch`, `torchvision`, `numpy`, `matplotlib`, `wandb` (`v0.28.1`)

### Installation
```bash
git clone https://github.com/cs25m504/mobilenetv2.git
cd mobilenetv2
pip install -r requirements.txt
```

To enable Weights & Biases logging during evaluation:
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

---

## 3. Reproducing Baseline Results (Question 1)

The baseline MobileNet-v2 modifies the stem convolution stride (`2 -> 1`) and first bottleneck stage stride (`2 -> 1`) so that $32 \times 32$ CIFAR-10 inputs preserve a $4 \times 4$ feature map prior to global average pooling.

### Run Baseline Training
```bash
python train.py --epochs 60 --batch_size 128 --lr 0.1 --weight_decay 5e-4 --seed 42
```

### Baseline Specifications & Metrics
* **Trainable Parameters:** `2,236,682` (`+ 9,920` BatchNorm buffer entries)
* **Baseline In-Memory Weight + Buffer Size:** `8.663 MB` (`8.76 MB` serialized `.pth` on disk)
* **Baseline FP32 Activation Memory (`1x3x32x32`):** `12.512 MB`
* **Best Test Top-1 Accuracy (60 Epochs):** **`93.55%`**
* **Outputs Saved:** `./checkpoints/baseline_mobilenetv2.pth` and `./checkpoints/baseline_curves.png`

---

## 4. Reproducing Compression Experiments (Questions 2–4)

### Run a Single Compression Configuration
Use `test.py` with `--weight_quant_bits` and `--activation_quant_bits`:
```bash
python test.py --weight_quant_bits 4 --activation_quant_bits 8
```

### Run the Full Compression Sweep
```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
python test.py --weight_quant_bits 8 --activation_quant_bits 4
python test.py --weight_quant_bits 4 --activation_quant_bits 8
python test.py --weight_quant_bits 4 --activation_quant_bits 4
python test.py --weight_quant_bits 4 --activation_quant_bits 3
python test.py --weight_quant_bits 3 --activation_quant_bits 4
python test.py --weight_quant_bits 3 --activation_quant_bits 3
python test.py --weight_quant_bits 2 --activation_quant_bits 4
```

---

## 5. Summary of Results

| Run Name | Weight Bits | Act Bits | Compressed Weight Size (MB) | Weight CR | Metadata Overhead (MB) | Compressed Act Size (MB) | Act CR | Overall CR | Top-1 Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w8_a8` | 8 | 8 | 2.447 | 3.54$\times$ | 0.0488 | 7.803 | 1.60$\times$ | 2.07$\times$ | **93.24%** |
| `w8_a4` | 8 | 4 | 2.447 | 3.54$\times$ | 0.0488 | 7.031 | 1.78$\times$ | 2.23$\times$ | 83.31% |
| `w4_a8` *(Optimal)* | 4 | 8 | 1.403 | 6.17$\times$ | 0.0488 | 7.803 | 1.60$\times$ | 2.30$\times$ | **91.77%** |
| `w4_a4` | 4 | 4 | 1.403 | 6.17$\times$ | 0.0488 | 7.031 | 1.78$\times$ | 2.51$\times$ | 83.08% |
| `w4_a3` | 4 | 3 | 1.403 | 6.17$\times$ | 0.0488 | 6.838 | 1.83$\times$ | 2.57$\times$ | 31.42% |
| `w3_a4` | 3 | 4 | 1.403 | 6.17$\times$ | 0.0488 | 7.031 | 1.78$\times$ | 2.51$\times$ | 75.00% |
| `w3_a3` | 3 | 3 | 1.403 | 6.17$\times$ | 0.0488 | 6.838 | 1.83$\times$ | 2.57$\times$ | 30.37% |
| `w2_a4` *(Max CR)* | 2 | 4 | **0.881** | **9.83$\times$** | 0.0488 | 7.031 | 1.78$\times$ | **2.68$\times$** | 11.32% |

# ISIC 2024 – Skin Cancer Detection Using Deep Learning & 3D (TBP)
Simple, step-by-step notebooks for building image and multi-modal models on the **ISIC 2024** dataset.  
You can watch the project walkthrough on YouTube here:  
[https://youtu.be/2v2x6x1gqqY](https://youtu.be/2v2x6x1gqqY)
---

##  What this project does

- Loads and explores the ISIC 2024 skin lesion dataset (HDF5 + metadata).
- Trains image classifiers with **ResNet50** (Fastai) and **EfficientFormerV2** backbones.
- Experiments with **oversampling** to handle class imbalance.
- Combines **images + tabular/meta features** in a single model (PyTorch notebook).
- Produces a `submission.csv` ready for competition submission or evaluation.

> All training is in Jupyter notebooks. No custom CLI is required.

---

##  Repository layout (high level)

```
Fastai/
  10k_to1k_2_models.ipynb           # Experiments scaling from 10k → 1k; two-model comparison
  No_oversampling_Resnet50.ipynb    # Baseline ResNet50 without oversampling
  Oversampling_only.ipynb           # ResNet50 with oversampling strategy
  efficientformerv2_s2_weights.pth  # Pretrained EfficientFormerV2-S2 weights (used in experiments)
  resnet50-11ad3fa6.pth             # Pretrained ResNet50 weights (ImageNet)

pytorch/
  Deep Learning_Images.ipynb        # Pure image model training (PyTorch)
  ISIC_inference.ipynb              # Inference pipeline to create submission.csv
  Image+MetaCombined.ipynb          # Multi‑modal (image + metadata) model

submission.csv                       # Example/last generated submission
.gitignore
ADA_1.ipynb
ADA_2.ipynb
ADA_3.ipynb
```

---

##  Data & File Management

### 1. Download from Kaggle (required)
The following files come directly from the **ISIC 2024 competition** and must be downloaded manually (we cannot redistribute them in GitHub):

```
/isic/image/                  
/isic/train-image.hdf5
/isic/train-metadata.csv
/isic/test-image.hdf5
/isic/test-metadata.csv
```

 Place them under `./data/ISIC2024/` (or update paths in notebooks).  
Your folder might look like this:

```
data/
  ISIC2024/
    image/
    train-image.hdf5
    test-image.hdf5
    train-metadata.csv
    test-metadata.csv
```

---

### 2. Files generated automatically ( NOT upload to GitHub)
When you run the notebooks, extra helper/processed files will be created.
```
/isic/models/                  # trained model checkpoints
/isic/Oversampled_small.hdf5
/isic/Small_df5.csv
/isic/new_cat_columns.csv      # unless manually created and required
/isic/sample_submission.csv    # Kaggle already provides one
```

 Add these to `.gitignore` so they don’t clutter the repo.

Example `.gitignore` rules:
```
# Large / generated files
*.hdf5
*.csv
models/
```

---

### 3. What stays in GitHub 
- All notebooks (`.ipynb`)  
- Pretrained weights (`.pth`) that are small enough  
- Config files created manually (if any)  
- `README.md` and `.gitignore`  

---

##  Prerequisites

- Python **3.10+**
- GPU is recommended (NVIDIA with CUDA). CPU will work but will be slow.
- Jupyter Lab / Jupyter Notebook

### Install dependencies

```bash
# create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# minimal dependencies
pip install --upgrade pip
# Choose the right torch build for your system (CUDA/CPU). The example below is CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fastai timm scikit-learn pandas numpy matplotlib h5py pillow tqdm
```

---

##  How to run

1. **Open Jupyter** in the repo root:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. **Pick a path** depending on what you want to try:

   - **Baseline (no oversampling)**  
     Open: `Fastai/No_oversampling_Resnet50.ipynb`  
     What you get: a clean ResNet50 baseline and training/validation metrics.

   - **Handle class imbalance (oversampling)**  
     Open: `Fastai/Oversampling_only.ipynb`  
     What you get: same model but with oversampling to improve minority class recall.

   - **Scale experiments (10k → 1k, two models)**  
     Open: `Fastai/10k_to1k_2_models.ipynb`  
     What you get: controlled down-sampling and a side-by-side model comparison.

   - **Pure PyTorch image model**  
     Open: `pytorch/Deep Learning_Images.ipynb`  
     What you get: a from-scratch PyTorch training loop for images only.

   - **Image + Metadata (multi-modal)**  
     Open: `pytorch/Image+MetaCombined.ipynb`  
     What you get: a combined model that ingests both image tensors and tabular features.

   - **Inference / Submission**  
     Open: `pytorch/ISIC_inference.ipynb`  
     What you get: loads a saved checkpoint and outputs `submission.csv`.

---

##  Methods summary

- **Backbones:** ResNet50 and EfficientFormerV2-S2 with ImageNet pretraining.  
- **Augmentations:** Standard image transforms in Fastai/PyTorch (flip, rotate, etc.).  
- **Imbalance handling:** Oversampling of minority classes (Fastai sampler).  
- **Optimization:** Adam/SGD (as configured per notebook), cosine/step LR schedulers.  
- **Evaluation:** Accuracy, F1, and confusion matrix (notebook cells).  
- **Multi-modal:** Concatenates CNN image embeddings with normalized tabular features before the classifier head.

---

## 📊 Results

| Model / Setting                     | Val Accuracy | F1 (macro) | Notes |
|------------------------------------|--------------|------------|-------|
| ResNet50 (no oversampling)         | ~0.89        | ~0.88      | baseline |
| ResNet50 (with oversampling)       | ~0.90        | ~0.89      | improved recall on minority classes |
| EfficientFormerV2-S2               | ~0.90        | ~0.89      | fast/compact backbone |
| Image + Meta (combined)            | ~0.91        | ~0.90      | best trade-off |

---

##  Reproducibility

- Set random seeds inside each notebook where provided.  
- Note that data loader order and CUDA kernels can still introduce minor nondeterminism.  
- Record the **PyTorch**, **CUDA**, and **driver** versions when reporting results.

---

##  Tips & common pitfalls

- Ensure the **data paths** in each notebook match your local layout.  
- If memory is tight, reduce image size or batch size first.  
- For Kaggle/ISIC submissions, verify the **column names and order** in `submission.csv` before uploading.

---

##  Acknowledgments

- ISIC Archive & ISIC 2024 Challenge organizers  
- Fastai and PyTorch communities  
- Pretrained backbones: torchvision and TIMM

---

##  License

This repository is for academic use.
---

##  Authors

- Yashwanth Jamalla (primary notebooks)

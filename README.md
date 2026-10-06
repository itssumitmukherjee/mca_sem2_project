# Soil Analysis & Crop Recommendation (PyTorch)

A neural network that recommends one of 22 crops from soil nutrients (N, P, K), pH, temperature, humidity and rainfall, plus a short soil report comparing your values with the recommended crop's typical range.

## Contents

| File | Purpose |
|---|---|
| `Crop_Recommendation_PyTorch.ipynb` | Full walkthrough with outputs: EDA, training, evaluation, inference |
| `crop_model.py` | Model, data prep, training loop, save/load, recommendation, soil report |
| `train.py` | Command-line training; writes `artifacts/` |
| `predict.py` | Command-line prediction for one sample |
| `data/Crop_recommendation.csv` | Dataset (2,200 rows, 22 crops × 100 samples) |
| `artifacts/` | Trained `crop_model.pt`, `metrics.json`, training curves, confusion matrix |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Use

```bash
python train.py                      # retrain (about a minute on CPU)
python predict.py --N 90 --P 42 --K 43 --temperature 21 --humidity 82 --ph 6.5 --rainfall 203
jupyter notebook Crop_Recommendation_PyTorch.ipynb
```

Run all commands from this folder.

## Model

MLP `7 → 128 → 64 → 22` with BatchNorm, ReLU and Dropout(0.2). Trained with AdamW, label smoothing 0.05, ReduceLROnPlateau and early stopping. Stratified 70/15/15 split; the scaler is fit on the training split only.

## Results (test set, 330 samples)

Accuracy **99.1%**, macro-F1 **0.991**. A random-forest baseline on the same split scores about 99.4%, so the dataset is easy for any reasonable model.

## Limitations

The dataset is small and clean; real fields are noisier. The model reflects typical conditions in the training data and ignores prices, rotation, soil texture and local practice. Treat it as decision support alongside a proper soil test.

# Ablation Study: DMAG-LSTM and BiLSTM Models

This folder contains the script `train_ablations.py`, which is used to conduct ablation experiments for website fingerprinting using different neural network architectures and feature combinations. The experiments are designed to analyze the impact of time-only features and the addition of statistical features on model performance.

## Script Overview

- **train_ablations.py**
  - Implements two main ablation experiments:
    - **ExpA_time_only_bidmag**: Trains a Bidirectional DMAG-LSTM model using only time sequence features.
    - **ExpB_vanilla_bilstm**: Trains a vanilla Bidirectional LSTM model with both time sequence and statistical features.
  - Supports flexible scope selection (e.g., top100, top200, all classes).
  - Automatically handles feature extraction, model building, training, fine-tuning, evaluation, and result saving.

## Common Parameters

- `--data`: Path to the input `.npz` data file containing features and labels
- `--outdir`: Output root directory for saving models and results
- `--scopes`: Class scope selection (e.g., top100,top200,all)
- `--epochs_base`: Number of epochs for base training
- `--epochs_ft`: Number of epochs for fine-tuning

## Outputs

- Trained model files (base and fine-tuned)
- Predictions and probability scores (CSV)
- Metrics summary (JSON)
- Aggregated results in Excel format (`ablation_metrics.xlsx`)

## Usage

Run the script with desired parameters, for example:
```bash
python train_ablations.py --data interval_time_closed_world_449.npz --outdir results --scopes top100,all --epochs_base 150 --epochs_ft 50
```

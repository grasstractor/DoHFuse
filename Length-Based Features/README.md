# Length-Based Feature Model Training Scripts

This folder contains Python scripts for training and evaluating machine learning models using packet length-based features for website fingerprinting. Each script corresponds to a different model architecture and approach.

## File Overview

- **train_cnn_pktlen.py**
  - Trains a 1D Convolutional Neural Network (CNN) using signed packet length sequences and mask features.
  - Handles feature extraction, model building, training, and evaluation.

- **train_lstm_pktlen.py**
  - Trains an LSTM-based neural network using packet length features.
  - Covers model definition, training, and performance metrics.

- **train_rf_pktlen.py**
  - Implements Random Forest classification using packet length features.
  - Includes feature extraction, model training, and evaluation.

## Common Parameters

- `--data`: Path to the input `.npz` data file containing features and labels
- `--epochs`: Number of training epochs (for neural network models)
- `--batch`: Batch size (for neural network models)
- `--lr`: Learning rate (for neural network models)
- `--model_out`: Path to save the trained model
- `--n_estimators`: Number of trees in the forest (for Random Forest)
- `--max_depth`: Maximum tree depth (for Random Forest)

## Outputs

- Trained model files
- Evaluation metrics and predictions

## Usage

Run each script independently with the desired parameters. Refer to the comments and argument parsers within each script for specific usage instructions and configurable options.

## Notes

- All scripts use packet length-based features for website fingerprinting classification.

---
For further details, see the code comments in each script or contact the project maintainer.

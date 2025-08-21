# Close-World Model Training Scripts

This folder contains Python scripts for training and evaluating various machine learning models on website fingerprinting data in a close-world setting. Each script corresponds to a different model or approach. Below is a brief description of each file:

## File Overview

- **train_dmaglstm.py**
  - Trains a Multi-scale Adaptive Gated LSTM (DMAG-LSTM) model for sequence classification.
  - Handles feature extraction, model building, training, and evaluation.
  - **Common Parameters:**
    - `--data`: Path to the input `.npz` data file
    - `--epochs`: Number of training epochs
    - `--batch`: Batch size
    - `--lr`: Learning rate
    - `--model_out`: Path to save the trained model

- **train_knn.py**
  - Implements K-Nearest Neighbors (KNN) classification for website fingerprinting.
  - Includes data loading, preprocessing, training, and test evaluation.
  - **Common Parameters:**
    - `--data`: Path to the input `.npz` data file
    - `--k`: Number of neighbors
    - `--metric`: Distance metric (e.g., 'euclidean')

- **train_lstm.py**
  - Trains a standard LSTM-based neural network for sequence classification.
  - Covers model definition, training, and performance metrics.
  - **Common Parameters:**
    - `--data`: Path to the input `.npz` data file
    - `--epochs`: Number of training epochs
    - `--batch`: Batch size
    - `--lr`: Learning rate
    - `--model_out`: Path to save the trained model

- **train_odoh.py**
  - Trains a Fully Connected Neural Network (FCNN) and Gated Recurrent Unit (GRU) ensemble model.
  - The methodology is adapted from the paper "Privacy Analysis of Oblivious DNS over HTTPS: a Website Fingerprinting Study".
  - **Common Parameters:**
    - `--data`: Path to the input `.npz` data file
    - `--epochs`: Number of training epochs
    - `--batch`: Batch size
    - `--lr`: Learning rate
    - `--model_out`: Path to save the trained model

- **train_rf.py**
  - Implements Random Forest classification for website fingerprinting.
  - Includes feature extraction, model training, and evaluation.
  - **Common Parameters:**
    - `--data`: Path to the input `.npz` data file
    - `--n_estimators`: Number of trees in the forest
    - `--max_depth`: Maximum tree depth
    - `--model_out`: Path to save the trained model

- **train_transformer.py**
  - Trains a Transformer-based neural network for sequence classification.
  - Handles model architecture, training, and evaluation.
  - **Common Parameters:**
    - `--data`: Path to the input `.npz` data file
    - `--epochs`: Number of training epochs
    - `--batch`: Batch size
    - `--lr`: Learning rate
    - `--model_out`: Path to save the trained model
    - `--num_layers`: Number of transformer layers
    - `--d_model`: Model dimension
    - `--num_heads`: Number of attention heads

## Usage

Each script is designed to be run independently. Please refer to the comments and argument parsers within each script for specific usage instructions, required data formats, and configurable parameters.

## Notes

- All scripts are intended for close-world experiments, where all classes are known during training and testing.
- For open-world or ablation studies, refer to the corresponding folders and scripts in the main project directory.

---
For further details, see the code comments in each script or contact the project maintainer.

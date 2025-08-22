# DOHFUSE: Website Fingerprinting Research Codebase

This repository contains the complete code used in our paper **"DOHFUSE: Website Fingerprinting Attacks and Defenses for Oblivious DNS over HTTPS"**. The codebase supports feature extraction, model training, ablation studies, open-world evaluation, and large-scale data collection for website fingerprinting experiments.

## Experimental Environment

- **OS:** Ubuntu 24.04
- **GPU:** NVIDIA A10
- **CPU:** 16 cores
- **Memory:** 60 GB RAM
- **Database:** All raw and processed data have been uploaded to ScienceDB. For perfect reproduction, please download the full database from ScienceDB. We also provide processed `.npz` files for convenience.

## Software Requirements

- **Python:** 3.13.3 (recommended)
- **Go:** 1.23 (for pcap analysis)
- **Python dependencies:** Install via `pip install -r requirements.txt`

## Folder Structure & Purpose

- **feature extraction/**
  - Scripts for extracting features from raw network traffic (interval time, packet length, ODoH-specific features).
  - Use these scripts to process CSV data (exported from pcap analysis) into `.npz` files for model training.

- **close-world/**
  - Model training scripts for close-world experiments (all classes known).
  - Includes CNN, LSTM, Transformer, Random Forest, KNN, and DMAG-LSTM models.
  - Each script supports flexible configuration and outputs trained models and metrics.

- **open-world/**
  - Scripts for open-world evaluation, including DMAG-LSTM model training and open-set recognition (temperature scaling, MSP thresholding).
  - Includes evaluation scripts for ROC/PR curve generation and threshold selection.

- **Length-Based Features/**
  - Model training scripts using packet length-based features (CNN, LSTM, Random Forest).
  - Designed for experiments focusing on length-only features.

- **abalations/**
  - Ablation study scripts to analyze the impact of different features and model architectures.
  - Includes DMAG-LSTM and vanilla BiLSTM experiments with/without statistical features.

- **selenium/**
  - Automated website visit and traffic capture scripts using Selenium and Chrome with DoH enabled.
  - Captures pcap files and SSL key logs for large-scale data collection.

- **pcap analysis/**
  - Go scripts for parsing and converting pcap files to CSV format, extracting packet-level information.

## Data & Reproducibility

- **Raw Data:** Download from Zenodo (link provided in the paper and supplementary materials).
- **Processed Data:** Pre-processed `.npz` files are also available on Zenodo for direct use in model training and evaluation.

## Reproduction Steps

1. **Install dependencies:**
   - Install Python 3.13.3 and Go 1.23.
   - Run `pip install -r requirements.txt` to install Python dependencies.
   - Run `go mod tidy` to install go dependencies.
2. **Download the database from Zenodo.**
3. **Convert raw pcap files to CSV:**
   - Use the scripts in `pcap analysis/pcaptocsv.go` to export CSV files from pcap data.
4. **Extract features:**
   - Use the scripts in `feature extraction/` to process CSV files and generate `.npz` feature files.
   - Alternatively, use the provided processed `.npz` files.
5. **Run experiments:**
   - Use the training scripts in each folder to reproduce the experiments and results.

## Notes

- All scripts are documented with usage instructions and parameter descriptions in their respective README files.
- For any issues or questions, please refer to the code comments or contact the authors.

---
**This codebase is provided for academic research and reproducibility of the DOHFUSE paper.**

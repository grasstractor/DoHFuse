# Open-World Evaluation Script

This script (`eval_openworld.py`) is used to evaluate open-world website fingerprinting models, particularly those trained with DMAG-LSTM architectures. It computes detection scores, generates ROC and PR curves, and determines the best threshold for distinguishing monitored (known) and unmonitored (unknown) traffic.

## Script Overview

- Loads a trained model and test datasets (closed-world and open-world).
- Computes scores for monitored and unmonitored samples using MSP, margin, or negative entropy.
- Concatenates results for binary classification (monitor=1, unmonitor=0).
- Generates ROC and Precision-Recall curves, saves them as images and CSV files.
- Finds the best threshold for F1 score and outputs summary metrics.
- Saves per-sample scores and summary results to the output directory.

## Common Parameters

- `--model`: Path to the trained `.keras` model (fine-tuned model recommended)
- `--closed_npz`: Path to closed-world `.npz` file (for monitored test data)
- `--open_npz`: Path to open-world `.npz` file (for unmonitored test data)
- `--outdir`: Output directory for evaluation results
- `--score`: Scoring method (`msp`, `margin`, or `neg_entropy`)

## Outputs

- ROC and PR curve images (`roc.png`, `pr_annotated.png`)
- ROC and PR curve data (CSV)
- Per-sample scores (`samples_scores.csv`)
- Summary metrics and best threshold (`open_eval_summary.json`)

## Usage

Example command:
```bash
python eval_openworld.py --model fine_tuned_model_DMAGLSTM_openworld.keras --closed_npz closed_world.npz --open_npz open_world.npz --outdir open_eval --score msp
```

## Notes

- The script is compatible with custom DMAGLSTMCell implementations (ensure the class is registered/imported).
- Supports multiple scoring methods for open-set detection.
- For more details, refer to the code comments in `eval_openworld.py` or contact the project maintainer.

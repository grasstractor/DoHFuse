# eval_open_world.py
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- Compatible with your custom DMAGLSTMCell (must be registered/imported with the same class name in the import path) ----
# If in the same directory as the training script, use: from bidmaglstm_scopes import DMAGLSTMCell
from train_dmaglstm import DMAGLSTMCell  # 确保路径正确

def load_closed_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["time_test"], d["stat_test"]

def load_open_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    # Compatible with two naming conventions
    time_key = "time_open" if "time_open" in d.files else "time_test"
    stat_key = "stat_open" if "stat_open" in d.files else "stat_test"
    y_open = d[ "y_open_monitored" ] if "y_open_monitored" in d.files else None
    return d[time_key], d[stat_key], y_open

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def predict_scores(model, time_arr, stat_arr, batch_size=1024):
    # Model input shapes: (N, L, 1) and (N, D)
    X_time = time_arr[..., np.newaxis]
    logits = model.predict([X_time, stat_arr], batch_size=batch_size, verbose=0)
    probs = softmax(logits, axis=1)
    max_prob = probs.max(axis=1)
    y_hat = probs.argmax(axis=1)
    return max_prob, y_hat, probs

def sweep_threshold(y_true_bin, score, outdir):
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

    os.makedirs(outdir, exist_ok=True)

    # --- ROC / PR calculation ---
    auroc = roc_auc_score(y_true_bin, score)
    fpr, tpr, _roc_thr = roc_curve(y_true_bin, score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_bin, score)
    auprc = average_precision_score(y_true_bin, score)

    # Calculate F1 at each point
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1s))  # precision/recall与f1等长；threshold少1个点

    # Align with threshold (precision/recall has one more starting point (1, P0))
    if best_idx == 0:
        best_thr = pr_thresholds[0]  # 取第一个阈值做近似
    else:
        best_thr = pr_thresholds[best_idx - 1]

    best = {
        "best_F1": float(f1s[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_threshold": float(best_thr)
    }

    # --- Save CSV (including thresholds) ---
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(outdir, "roc_curve.csv"), index=False)
    # Align PR thresholds with points: thresholds correspond to the last N-1 points of (precision, recall)
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
    # For alignment, the first row threshold has no correspondence, set to NaN, subsequent rows align with thresholds
        "threshold": np.concatenate(([np.nan], pr_thresholds))
    })
    pr_df.to_csv(os.path.join(outdir, "pr_curve_with_thresholds.csv"), index=False)

    # --- Plot ROC ---
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=1.8, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], ls="--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Open-World)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc.png"), dpi=150)
    plt.close()

    # --- Plot PR + mark best threshold point ---
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=1.8, label=f"AUPRC = {auprc:.4f}")
    # Mark the best point
    bx, by = recall[best_idx], precision[best_idx]
    plt.scatter([bx], [by], s=60, color="red", zorder=5)
    # Text label
    txt = (f"Best F1 = {f1s[best_idx]:.4f}\n"
           f"P = {precision[best_idx]:.4f}, R = {recall[best_idx]:.4f}\n"
           f"Thr = {best_thr:.4f}")
    # Move text slightly to avoid covering the point
    plt.annotate(txt, xy=(bx, by), xytext=(bx*0.85, min(by+0.08, 1.0)),
                 arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (Open-World) with Best-Threshold Marker")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_annotated.png"), dpi=150)
    plt.close()

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        **best
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the trained .keras model (fine-tuned model recommended)")
    ap.add_argument("--closed_npz", required=True, help="Closed-world .npz file used for training (for time_test/stat_test)")
    ap.add_argument("--open_npz", required=True, help="Open-world .npz file (time_open/stat_open or time_test/stat_test)")
    ap.add_argument("--outdir", default="open_eval", help="Output directory for evaluation results")
    ap.add_argument("--score", default="msp", choices=["msp","margin","neg_entropy"], help="Open-world scoring method")
    args = ap.parse_args()

    # Load model
    model = load_model(args.model, compile=False)

    # Load data
    time_m, stat_m = load_closed_npz(args.closed_npz)   # monitor = closed-world test
    time_u, stat_u, y_open = load_open_npz(args.open_npz)  # unmonitor = open-world

    # Predict scores
    score_m, yhat_m, probs_m = predict_scores(model, time_m, stat_m)
    score_u, yhat_u, probs_u = predict_scores(model, time_u, stat_u)

    if args.score == "margin":
        # top1 - top2
        def margin(probs):
            part = -np.sort(-probs, axis=1)  # descending order
            return part[:,0] - part[:,1]
        score_m = margin(probs_m); score_u = margin(probs_u)
    elif args.score == "neg_entropy":
        def entropy(p): return -np.sum(p*np.log(np.clip(p,1e-12,1.0)), axis=1)
        score_m = -entropy(probs_m); score_u = -entropy(probs_u)
    # else MSP is already handled by default

    # Concatenate as binary evaluation set: monitor=1, unmonitor=0
    y_true_bin = np.concatenate([np.ones_like(score_m, dtype=int), np.zeros_like(score_u, dtype=int)])
    score_all  = np.concatenate([score_m, score_u])

    # Export per-sample results
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame({
        "set": ["monitor"]*len(score_m) + ["unmonitor"]*len(score_u),
        "score": score_all
    }).to_csv(os.path.join(args.outdir, "samples_scores.csv"), index=False)

    # Curves and best threshold
    summary = sweep_threshold(y_true_bin, score_all, args.outdir)

    # Give overall F1 at the best threshold
    thr = summary["best_threshold"]
    y_pred_bin = (score_all >= thr).astype(int)
    f1_overall = f1_score(y_true_bin, y_pred_bin)
    summary["f1_at_best_threshold"] = float(f1_overall)
    summary["threshold_used"] = float(thr)

    with open(os.path.join(args.outdir, "open_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Open-World Eval Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

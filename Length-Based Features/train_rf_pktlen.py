# rf_train_eval.py
import os, time, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, top_k_accuracy_score,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def four_metrics(y_true, y_pred, average: str = "macro"):
    """
    Only calculate and return four metrics:
      - Accuracy
      - Precision (macro/micro/weighted, default macro)
      - Recall    (macro/micro/weighted, default macro)
      - F1Score   (macro/micro/weighted, default macro)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

def print_four_metrics(name: str, acc, prec, rec, f1):
    print(f"[{name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1Score: {f1:.4f}")

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    def get(k): 
        if k not in data.files:
            raise KeyError(f"Missing key '{k}' in {npz_path}")
        return data[k]
    return {
        "Xtr": get("time_train"),
        "Xte": get("time_test"),
        "ytr": get("y_train"),
        "yte": get("y_test"),
        "classes": get("classes"),
        "train_paths": data["train_paths"] if "train_paths" in data.files else None,
        "test_paths":  data["test_paths"]  if "test_paths"  in data.files else None,
    }

def build_features(X):
    mask = (X != -1).astype(np.float32)
    X0 = np.where(X == -1, 0.0, X).astype(np.float32)
    feats = np.concatenate([X0, mask], axis=1)
    return feats

def plot_confusion(y_true, y_pred, class_names, outdir, fname, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize=('true' if normalize else None))
    plt.figure(figsize=(7.5, 6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=180); plt.close()
    pd.DataFrame(cm, index=class_names, columns=class_names)\
      .to_csv(os.path.join(outdir, fname.replace('.png', '.csv')))

def compute_roc_pr(probs, y_true, n_classes, outdir):
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    # micro ROC
    fpr, tpr, _ = roc_curve(Y.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'micro-AUC={roc_auc_micro:.3f}')
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (micro)'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'roc_micro.png'), dpi=180); plt.close()
    # micro PR
    prec, rec, _ = precision_recall_curve(Y.ravel(), probs.ravel())
    ap_micro = average_precision_score(Y, probs, average='micro')
    plt.figure(); plt.plot(rec, prec, label=f'micro-AP={ap_micro:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR (micro)'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'pr_micro.png'), dpi=180); plt.close()
    # per-class AUC/AP + macro
    roc_auc_dict, ap_dict = {}, {}
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(Y[:, i], probs[:, i])
        roc_auc_dict[i] = auc(fpr_i, tpr_i)
        prec_i, rec_i, _ = precision_recall_curve(Y[:, i], probs[:, i])
        ap_dict[i] = average_precision_score(Y[:, i], probs[:, i])
    macro_auc = float(np.mean(list(roc_auc_dict.values())))
    macro_ap  = float(np.mean(list(ap_dict.values())))
    with open(os.path.join(outdir, 'roc_summary.json'), "w", encoding="utf-8") as f:
        json.dump({"micro_auc": float(roc_auc_micro), "macro_auc": macro_auc,
                   "per_class_auc": {str(k): float(v) for k, v in roc_auc_dict.items()}},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, 'pr_summary.json'), "w", encoding="utf-8") as f:
        json.dump({"micro_ap": float(ap_micro), "macro_ap": macro_ap,
                   "per_class_ap": {str(k): float(v) for k, v in ap_dict.items()}},
                  f, ensure_ascii=False, indent=2)
    pd.DataFrame({"class_id": np.arange(n_classes),
                  "auc": [roc_auc_dict[i] for i in range(n_classes)]})\
      .to_csv(os.path.join(outdir, 'per_class_auc.csv'), index=False)
    pd.DataFrame({"class_id": np.arange(n_classes),
                  "ap":  [ap_dict[i] for i in range(n_classes)]})\
      .to_csv(os.path.join(outdir, 'per_class_ap.csv'), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="packet_length_signed_449_all.npz",
                    help=".npz path (contains time_train/time_test etc.)")
    ap.add_argument("--outdir", default=None, help="output directory (default rf_reports/timestamp)")
    ap.add_argument("--model_out", default=None, help="model save path (default rf_model.pkl in output dir)")
    ap.add_argument("--trees", type=int, default=100, help="number of random forest trees")
    ap.add_argument("--max_depth", type=int, default=12, help="max depth (None=unlimited)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("rf_reports", ts)
    ensure_dir(outdir)
    model_out = args.model_out or os.path.join(outdir, "rf_model.pkl")

    # 1) Load data
    data = load_npz(args.data)
    Xtr_raw, Xte_raw = data["Xtr"], data["Xte"]
    ytr, yte = data["ytr"], data["yte"]
    class_names = [str(c) for c in data["classes"]]
    n_classes = len(class_names)
    seq_len = Xtr_raw.shape[1]

    # 2) Build features (set value to 0 + mask)
    Xtr = build_features(Xtr_raw)  # (N, 2*L)
    Xte = build_features(Xte_raw)

    # 3) Train random forest (only save final model)
    rf = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )
    rf.fit(Xtr, ytr)

    # 4) Evaluation
    y_pred = rf.predict(Xte)

    acc, prec, rec, f1 = four_metrics(yte, y_pred, average="macro")
    print_four_metrics("RandomForest", acc, prec, rec, f1)


if __name__ == "__main__":
    main()

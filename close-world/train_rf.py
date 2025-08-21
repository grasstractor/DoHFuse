# rf_raw_timeseries_paperparams.py
import os, argparse, json
import numpy as np
import pandas as pd
from joblib import dump
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ===== Load .npz (only use time series branch) =====
def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    need = ["time_train","time_test","y_train","y_test","classes"]
    for k in need:
        if k not in data.files:
            raise KeyError(f"Missing '{k}' in {npz_path}")
    Xtr = data["time_train"].astype(np.float32)   # (Ntr, L)
    Xte = data["time_test"].astype(np.float32)    # (Nte, L)
    ytr = data["y_train"].astype(int)
    yte = data["y_test"].astype(int)
    classes = [str(c) for c in data["classes"]]
    paths_te = data["test_paths"] if "test_paths" in data.files else None
    return Xtr, Xte, ytr, yte, classes, paths_te

# ===== Keep only the first N classes (or all) =====
def subset_first_n_classes(Xtr, Xte, ytr, yte, classes, n_first=None):
    if (n_first is None) or (n_first >= len(classes)):
        return Xtr, Xte, ytr, yte, classes, np.arange(len(classes))
    keep_names = set(classes[:n_first])
    keep_ids = np.array([i for i, name in enumerate(classes) if name in keep_names], dtype=int)
    tr_mask = np.isin(ytr, keep_ids)
    te_mask = np.isin(yte, keep_ids)
    Xtr_s, Xte_s = Xtr[tr_mask], Xte[te_mask]
    ytr_s, yte_s = ytr[tr_mask], yte[te_mask]
    # Old ID -> New ID (0..C'-1)
    id_map = {old:i for i, old in enumerate(sorted(keep_ids))}
    ytr_s = np.array([id_map[v] for v in ytr_s], dtype=int)
    yte_s = np.array([id_map[v] for v in yte_s], dtype=int)
    classes_s = [classes[old] for old in sorted(keep_ids)]
    return Xtr_s, Xte_s, ytr_s, yte_s, classes_s, np.array(sorted(keep_ids), dtype=int)

# ===== Metrics =====
def eval_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

# ===== Train and evaluate one scope =====
def run_scope(tag, n_first, Xtr0, Xte0, ytr0, yte0, classes0, args):
    # Select classes
    Xtr, Xte, ytr, yte, classes, kept_ids = subset_first_n_classes(
        Xtr0, Xte0, ytr0, yte0, classes0, n_first
    )
    scope_dir = os.path.join(args.outdir, f"{tag}_{len(classes)}cls")
    ensure_dir(scope_dir)

    # Only use raw time series: replace padding -1 with 0 (avoid treating as real value)
    Xtr_vec = np.where(Xtr == -1, 0.0, Xtr).astype(np.float32)
    Xte_vec = np.where(Xte == -1, 0.0, Xte).astype(np.float32)

    # Random Forest (parameters follow paper style: 70 trees + standard settings)
    rf = RandomForestClassifier(
        n_estimators=70,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(Xtr_vec, ytr)

    # Evaluation
    y_pred = rf.predict(Xte_vec)
    metrics = eval_metrics(yte, y_pred)

    # Probability (prob_top1)
    if hasattr(rf, "predict_proba"):
        probs = rf.predict_proba(Xte_vec)
        prob_top1 = probs.max(axis=1)
    else:
        prob_top1 = np.full_like(y_pred, np.nan, dtype=float)

    # Save model
    model_path = os.path.join(scope_dir, "rf_raw_timeseries.joblib")
    dump({
        "model": rf,
        "classes": classes,
        "kept_original_class_ids": kept_ids,
        "sequence_len": int(Xtr_vec.shape[1]),
        "pad_replaced_with": 0.0
    }, model_path)

    # Save prediction details
    pred_cols = {"y_true": yte, "y_pred": y_pred, "prob_top1": prob_top1}
    pred_df = pd.DataFrame(pred_cols)
    pred_df.to_csv(os.path.join(scope_dir, "predictions.csv"), index=False)

    # Save metrics JSON
    with open(os.path.join(scope_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scope": f"{tag}_{len(classes)}cls",
            "num_classes": len(classes),
            "train_samples": int(Xtr_vec.shape[0]),
            "test_samples": int(Xte_vec.shape[0]),
            "sequence_len": int(Xtr_vec.shape[1]),
            **{k: float(v) for k, v in metrics.items()},
        }, f, ensure_ascii=False, indent=2)

    # Summary row & class name table
    summary_row = {
        "scope": f"{tag}_{len(classes)}cls",
        "acc": metrics["acc"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "train_samples": Xtr_vec.shape[0],
        "test_samples": Xte_vec.shape[0],
        "num_classes": len(classes),
        "sequence_len": Xtr_vec.shape[1],
        "model_path": model_path
    }
    classes_df = pd.DataFrame({"class_id_new": np.arange(len(classes)), "class_name": classes})
    return summary_row, classes_df, scope_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="interval_time_closed_world_449.npz", help=".npz path")
    ap.add_argument("--outdir", default="rf_raw_runs", help="output root directory")
    ap.add_argument("--scopes", default="top100,top200,all", help="evaluation scopes: top100,top200,all (comma separated)")
    args = ap.parse_args()

    Xtr0, Xte0, ytr0, yte0, classes0, _ = load_npz(args.data)
    ensure_dir(args.outdir)

    # Scope parsing
    scopes_cfg = []
    for tag in [s.strip() for s in args.scopes.split(",") if s.strip()]:
        if tag.lower().startswith("top"):
            n = int(tag.lower().replace("top",""))
            scopes_cfg.append((tag, n))
        elif tag.lower() == "all":
            scopes_cfg.append((tag, None))
        else:
            raise ValueError(f"Unknown scope '{tag}'")

    summary_rows = []
    detail_sheets = {}

    for tag, n_first in scopes_cfg:
        row, df_detail, scope_dir = run_scope(
            tag, n_first, Xtr0, Xte0, ytr0, yte0, classes0, args
        )
        summary_rows.append(row)
        detail_sheets[f"classes_{tag}"] = df_detail

    # Write Excel summary
    xlsx_path = os.path.join(args.outdir, "rf_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for name, df in detail_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"[OK] Output directory: {args.outdir}")
    print(f"[OK] Summary Excel: {xlsx_path}")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n=== Summary ===")
        cols = ["scope","acc","f1_macro","train_samples","test_samples","num_classes","sequence_len"]
        print(df[cols])

if __name__ == "__main__":
    main()

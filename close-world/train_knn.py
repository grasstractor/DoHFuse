# knn_eval_timeseries_paper.py
import os, argparse, json
import numpy as np
import pandas as pd
from joblib import dump
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    need = ["time_train","time_test","y_train","y_test","classes"]
    for k in need:
        if k not in data.files:
            raise KeyError(f"Missing '{k}' in {npz_path}")
    Xtr = data["time_train"].astype(np.float32)
    Xte = data["time_test"].astype(np.float32)
    # -1 (padding) → 0 (avoid KNN distance interference from mask)
    Xtr = np.where(Xtr == -1, 0.0, Xtr)
    Xte = np.where(Xte == -1, 0.0, Xte)
    ytr = data["y_train"].astype(int)
    yte = data["y_test"].astype(int)
    classes = [str(c) for c in data["classes"]]
    return Xtr, Xte, ytr, yte, classes

def subset_first_n_classes(Xtr, Xte, ytr, yte, classes, n_first=None):
    if (n_first is None) or (n_first >= len(classes)):
        return Xtr, Xte, ytr, yte, classes, np.arange(len(classes))
    keep_names = set(classes[:n_first])
    keep_ids = np.array([i for i, name in enumerate(classes) if name in keep_names], dtype=int)
    tr_mask = np.isin(ytr, keep_ids)
    te_mask = np.isin(yte, keep_ids)
    Xtr_s, Xte_s = Xtr[tr_mask], Xte[te_mask]
    ytr_s, yte_s = ytr[tr_mask], yte[te_mask]
    # Old ID → New ID (0..C'-1)
    id_map = {old:i for i, old in enumerate(sorted(keep_ids))}
    ytr_s = np.array([id_map[v] for v in ytr_s], dtype=int)
    yte_s = np.array([id_map[v] for v in yte_s], dtype=int)
    classes_s = [classes[old] for old in sorted(keep_ids)]
    return Xtr_s, Xte_s, ytr_s, yte_s, classes_s, np.array(sorted(keep_ids), dtype=int)

def choose_k_cv(Xtr, ytr, k_grid, max_splits=10, random_state=42):
    # No more than the minimum class sample count
    min_class = min(Counter(ytr).values())
    n_splits = max(2, min(max_splits, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_k, best_score = None, -1.0
    for k in k_grid:
        f1s = []
        for tr_idx, va_idx in skf.split(Xtr, ytr):
            clf = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
            clf.fit(Xtr[tr_idx], ytr[tr_idx])
            pred = clf.predict(Xtr[va_idx])
            f1 = f1_score(ytr[va_idx], pred, average="macro", zero_division=0)
            f1s.append(f1)
        score = float(np.mean(f1s)) if f1s else -1.0
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score, n_splits

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

def run_scope(tag, n_first, Xtr0, Xte0, ytr0, yte0, classes0, force_k1, k_grid, out_root):
    Xtr, Xte, ytr, yte, classes, kept_ids = subset_first_n_classes(
        Xtr0, Xte0, ytr0, yte0, classes0, n_first
    )
    scope_dir = os.path.join(out_root, f"{tag}_{len(classes)}cls")
    ensure_dir(scope_dir)

    if force_k1:
        chosen_k, cv_f1, n_splits = 1, None, None
    else:
        chosen_k, cv_f1, n_splits = choose_k_cv(Xtr, ytr, k_grid, max_splits=10)

    clf = KNeighborsClassifier(n_neighbors=chosen_k, weights="uniform", metric="euclidean")
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    metrics = eval_metrics(yte, y_pred)

    model_path = os.path.join(scope_dir, f"knn_k{chosen_k}.joblib")
    dump({"model": clf, "classes": classes, "kept_original_class_ids": kept_ids}, model_path)

    pd.DataFrame({"y_true": yte, "y_pred": y_pred}).to_csv(
        os.path.join(scope_dir, "predictions.csv"), index=False
    )

    with open(os.path.join(scope_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scope": f"{tag}_{len(classes)}cls",
            "num_classes": len(classes),
            "train_samples": int(Xtr.shape[0]),
            "test_samples": int(Xte.shape[0]),
            "sequence_len": int(Xtr.shape[1]),
            "chosen_k": int(chosen_k),
            "cv_macro_f1": None if cv_f1 is None else float(cv_f1),
            "cv_n_splits": n_splits,
            **{k: float(v) for k, v in metrics.items()},
        }, f, ensure_ascii=False, indent=2)

    summary = {
        "scope": f"{tag}_{len(classes)}cls",
        "chosen_k": chosen_k,
        "cv_f1_macro": cv_f1,
        "acc": metrics["acc"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "train_samples": Xtr.shape[0],
        "test_samples": Xte.shape[0],
        "num_classes": len(classes),
        "sequence_len": Xtr.shape[1],
        "model_path": model_path,
    }
    classes_df = pd.DataFrame({"class_id_new": np.arange(len(classes)), "class_name": classes})
    return summary, classes_df, scope_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="interval_time_closed_world_449.npz", help=".npz path")
    ap.add_argument("--outdir", default="knn_runs_paper", help="output root directory")
    ap.add_argument("--scopes", default="top100,top200,all",
                    help="Evaluation scope: top100,top200,all (comma separated)")
    ap.add_argument("--force_k1", action="store_true", default=True,
                    help="Force k=1 as in the paper (enabled by default). To grid search, add --no-force_k1")
    ap.add_argument("--no-force_k1", dest="force_k1", action="store_false")
    ap.add_argument("--k_grid", default="1,3,5,7,9", help="k search grid (effective when force_k1 is off)")
    args = ap.parse_args()

    Xtr0, Xte0, ytr0, yte0, classes0 = load_npz(args.data)
    ensure_dir(args.outdir)
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]

    scopes_cfg = []
    for tag in [s.strip() for s in args.scopes.split(",") if s.strip()]:
        if tag.lower().startswith("top"):
            n = int(tag.lower().replace("top",""))
            scopes_cfg.append((tag, n))
        elif tag.lower() == "all":
            scopes_cfg.append((tag, None))
        else:
            raise ValueError(f"Unknown scope '{tag}'")

    summary_rows, detail_sheets = [], {}

    for tag, n_first in scopes_cfg:
        row, df_detail, scope_dir = run_scope(
            tag, n_first, Xtr0, Xte0, ytr0, yte0, classes0,
            force_k1=args.force_k1, k_grid=k_grid, out_root=args.outdir
        )
        summary_rows.append(row)
        detail_sheets[f"classes_{tag}"] = df_detail

    # Write Excel
    xlsx_path = os.path.join(args.outdir, "knn_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for name, df in detail_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"[OK] Output directory: {args.outdir}")
    print(f"[OK] Summary Excel: {xlsx_path}")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n=== Summary ===")
        cols = ["scope","chosen_k","acc","f1_macro","train_samples","test_samples","num_classes"]
        print(df[cols])

if __name__ == "__main__":
    main()

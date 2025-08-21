# lstm_raw_timeseries_paperparams.py
import os, argparse, json, random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical, set_random_seed

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); set_random_seed(SEED)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- Data loading (only use time series branch) ----------
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
    return Xtr, Xte, ytr, yte, classes

# ---------- Subset: first N classes or all ----------
def subset_first_n_classes(Xtr, Xte, ytr, yte, classes, n_first=None):
    if (n_first is None) or (n_first >= len(classes)):
        kept = np.arange(len(classes))
        return Xtr, Xte, ytr, yte, classes, kept
    keep_names = set(classes[:n_first])
    keep_ids = np.array([i for i, name in enumerate(classes) if name in keep_names], dtype=int)
    tr_mask = np.isin(ytr, keep_ids)
    te_mask = np.isin(yte, keep_ids)
    Xtr_s, Xte_s = Xtr[tr_mask], Xte[te_mask]
    ytr_s, yte_s = ytr[tr_mask], yte[te_mask]
    # Old id -> New id (0..C'-1)
    id_map = {old:i for i, old in enumerate(sorted(keep_ids))}
    ytr_s = np.array([id_map[v] for v in ytr_s], dtype=int)
    yte_s = np.array([id_map[v] for v in yte_s], dtype=int)
    classes_s = [classes[old] for old in sorted(keep_ids)]
    return Xtr_s, Xte_s, ytr_s, yte_s, classes_s, np.array(sorted(keep_ids), dtype=int)

# ---------- Model (paper-style two-layer LSTM-128 + Dropout 0.2) ----------
def build_lstm_model(seq_len, num_classes, units=128, drop=0.2, lr=1e-3):
    inp = layers.Input(shape=(seq_len, 1), name="ts")
    x = layers.Masking(mask_value=-1.0)(inp)               # Ignore padding -1
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(drop)(x)
    x = layers.LSTM(units, return_sequences=False)(x)
    x = layers.Dropout(drop)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------- Metrics ----------
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

# ---------- Train and evaluate one scope ----------
def run_scope(tag, n_first, Xtr0, Xte0, ytr0, yte0, classes0, args):
    Xtr, Xte, ytr, yte, classes, kept_ids = subset_first_n_classes(
        Xtr0, Xte0, ytr0, yte0, classes0, n_first
    )
    scope_dir = os.path.join(args.outdir, f"{tag}_{len(classes)}cls")
    ensure_dir(scope_dir)

    # Shape: (N, L, 1), keep -1 in original sequence, use with Masking
    L = Xtr.shape[1]
    x_tr = Xtr[..., np.newaxis]
    x_te = Xte[..., np.newaxis]

    # Train/validation split (stratified 10% as val; if fails, use random split)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        (tr_idx, va_idx) = next(sss.split(x_tr, ytr))
        x_tr_tr, x_tr_va = x_tr[tr_idx], x_tr[va_idx]
        y_tr_tr, y_tr_va = ytr[tr_idx], ytr[va_idx]
    except Exception:
        n = x_tr.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        cut = int(n * 0.9)
        tr_idx, va_idx = idx[:cut], idx[cut:]
        x_tr_tr, x_tr_va = x_tr[tr_idx], x_tr[va_idx]
        y_tr_tr, y_tr_va = ytr[tr_idx], ytr[va_idx]

    # one-hot
    y_tr_tr_oh = to_categorical(y_tr_tr, num_classes=len(classes))
    y_tr_va_oh = to_categorical(y_tr_va, num_classes=len(classes))
    y_te_oh    = to_categorical(yte,     num_classes=len(classes))

    # Class weights (more stable for imbalanced samples)
    cw = compute_class_weight(class_weight="balanced",
                              classes=np.unique(y_tr_tr),
                              y=y_tr_tr)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    # Model
    model = build_lstm_model(L, len(classes),
                             units=args.units, drop=args.dropout, lr=args.lr)

    # Callbacks
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                patience=args.patience, restore_best_weights=True)
    ]

    # Training
    hist = model.fit(
        x_tr_tr, y_tr_tr_oh,
        validation_data=(x_tr_va, y_tr_va_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=cbs
    )

    # Testing
    probs = model.predict(x_te, batch_size=args.batch_size, verbose=0)
    y_pred = probs.argmax(axis=1)
    metrics = eval_metrics(yte, y_pred)

    # Save model
    model_path = os.path.join(scope_dir, "lstm_raw_timeseries.keras")
    model.save(model_path)

    # Prediction details
    pd.DataFrame({
        "y_true": yte,
        "y_pred": y_pred,
        "prob_top1": probs.max(axis=1)
    }).to_csv(os.path.join(scope_dir, "predictions.csv"), index=False)

    # Metrics JSON
    with open(os.path.join(scope_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scope": f"{tag}_{len(classes)}cls",
            "num_classes": len(classes),
            "train_samples": int(x_tr.shape[0]),
            "val_samples": int(x_tr_va.shape[0]),
            "test_samples": int(x_te.shape[0]),
            "sequence_len": int(L),
            "units": int(args.units),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            **{k: float(v) for k, v in metrics.items()},
        }, f, ensure_ascii=False, indent=2)

    # Summary / class name mapping
    summary_row = {
        "scope": f"{tag}_{len(classes)}cls",
        "acc": metrics["acc"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "train_samples": x_tr.shape[0],
        "val_samples": x_tr_va.shape[0],
        "test_samples": x_te.shape[0],
        "num_classes": len(classes),
        "sequence_len": L,
        "model_path": model_path
    }
    classes_df = pd.DataFrame({"class_id_new": np.arange(len(classes)),
                               "class_name": classes})
    return summary_row, classes_df, scope_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="interval_time_closed_world_449.npz", help=".npz path")
    ap.add_argument("--outdir", default="lstm_raw_runs", help="output root directory")
    ap.add_argument("--scopes", default="top100,top200,all", help="top100,top200,all (comma separated)")

    # Hyperparameters (refer to paper: two-layer LSTM-128 + Dropout 0.2)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    Xtr0, Xte0, ytr0, yte0, classes0 = load_npz(args.data)
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

    summary_rows, detail_sheets = [], {}

    for tag, n_first in scopes_cfg:
        row, df_detail, scope_dir = run_scope(
            tag, n_first, Xtr0, Xte0, ytr0, yte0, classes0, args
        )
        summary_rows.append(row)
        detail_sheets[f"classes_{tag}"] = df_detail

    # Excel summary
    xlsx_path = os.path.join(args.outdir, "lstm_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for name, df in detail_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"[OK] Output directory: {args.outdir}")
    print(f"[OK] Summary Excel: {xlsx_path}")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n=== Summary ===")
        cols = ["scope","acc","f1_macro","train_samples","val_samples","test_samples","num_classes"]
        print(df[cols])

if __name__ == "__main__":
    main()

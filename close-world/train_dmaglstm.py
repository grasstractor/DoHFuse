# bidmaglstm_scopes.py
import os, json, argparse, random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout, BatchNormalization,
                                     GlobalAveragePooling1D, Masking, RNN, Bidirectional)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.saving import register_keras_serializable
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- Reproducibility ----------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- DMAG-LSTM Cell ----------------
@register_keras_serializable()
class DMAGLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]
    def build(self, input_shape):
        d = input_shape[-1]
        self.W_f_short = self.add_weight(shape=(d + self.units, self.units), initializer='glorot_uniform', name='W_f_short')
        self.b_f_short = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Constant(-1.0), name='b_f_short')
        self.W_f_long  = self.add_weight(shape=(d + self.units, self.units), initializer='glorot_uniform', name='W_f_long')
        self.b_f_long  = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Constant(1.0),  name='b_f_long')
        self.W_alpha   = self.add_weight(shape=(d + self.units, self.units), initializer='glorot_uniform', name='W_alpha')
        self.b_alpha   = self.add_weight(shape=(self.units,), initializer='zeros', name='b_alpha')
        self.W_m       = self.add_weight(shape=(d + self.units, d), initializer='glorot_uniform', name='W_m')
        self.b_m       = self.add_weight(shape=(d,), initializer='zeros', name='b_m')
        self.W_C       = self.add_weight(shape=(d + self.units, self.units), initializer='glorot_uniform', name='W_C')
        self.b_C       = self.add_weight(shape=(self.units,), initializer='zeros', name='b_C')
        self.W_o       = self.add_weight(shape=(d + self.units, self.units), initializer='glorot_uniform', name='W_o')
        self.b_o       = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')
        super().build(input_shape)
    def call(self, inputs, states):
        h_prev, c_prev = states
        hx = tf.concat([h_prev, inputs], axis=-1)
        f_short = tf.sigmoid(tf.matmul(hx, self.W_f_short) + self.b_f_short)
        f_long  = tf.sigmoid(tf.matmul(hx, self.W_f_long)  + self.b_f_long)
        alpha   = tf.sigmoid(tf.matmul(hx, self.W_alpha)   + self.b_alpha)
        f_t = alpha * f_short + (1.0 - alpha) * f_long
        i_t = 1.0 - f_t
        m_t = tf.sigmoid(tf.matmul(hx, self.W_m) + self.b_m)
        mod_x = m_t * inputs
        hxm   = tf.concat([h_prev, mod_x], axis=-1)
        c_hat = tf.tanh(tf.matmul(hxm, self.W_C) + self.b_C)
        c_t   = f_t * c_prev + i_t * c_hat
        o_t   = tf.sigmoid(tf.matmul(hx, self.W_o) + self.b_o)
        h_t   = o_t * tf.tanh(c_t)
        return h_t, [h_t, c_t]

# ---------------- Model ----------------
def build_base_model(sequence_length, num_stat_features, num_classes):
    time_input = Input(shape=(sequence_length, 1), name='time_input')
    stat_input = Input(shape=(num_stat_features,), name='stat_input')

    x = Masking(mask_value=-1)(time_input)
    x = Bidirectional(
            RNN(DMAGLSTMCell(128), return_sequences=True),
            merge_mode='concat',
            name='bi_dmag'
        )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)

    y = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(stat_input)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    merged = Concatenate()([x, y])
    z = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)

    out = Dense(num_classes)(z)  # logits
    return Model(inputs=[time_input, stat_input], outputs=out)

def lr_schedule(epoch):
    initial_lr = 0.001
    decay_factor = 0.9
    decay_epochs = 10
    return initial_lr * (decay_factor ** (epoch // decay_epochs))

# ---------------- Data ----------------
def load_data(npz_file):
    d = np.load(npz_file, allow_pickle=True)
    need = ["time_train","stat_train","y_train","time_test","stat_test","y_test","classes"]
    for k in need:
        if k not in d.files:
            raise KeyError(f"Missing '{k}' in {npz_file}")
    return {
        "time_train": d["time_train"],
        "stat_train": d["stat_train"],
        "y_train":    d["y_train"].astype(int),
        "time_test":  d["time_test"],
        "stat_test":  d["stat_test"],
        "y_test":     d["y_test"].astype(int),
        "classes":    [str(c) for c in d["classes"]],
    }

def subset_first_n_classes(data_dict, n_first=None):
    classes = data_dict["classes"]
    if (n_first is None) or (n_first >= len(classes)):
        kept_ids = np.arange(len(classes), dtype=int)
    else:
        kept_ids = np.arange(n_first, dtype=int)

    def _filter_split(split):
        y = data_dict[f"y_{split}"]
        m = np.isin(y, kept_ids)
        return {
            "time": data_dict[f"time_{split}"][m],
            "stat": data_dict[f"stat_{split}"][m],
            "y":    y[m]
        }

    tr_raw = _filter_split("train")
    te_raw = _filter_split("test")

    # Remap to 0..C'-1 for the first time
    id_map = {old:i for i, old in enumerate(kept_ids)}
    y_tr = np.array([id_map[v] for v in tr_raw["y"]], dtype=int)
    y_te = np.array([id_map[v] for v in te_raw["y"]], dtype=int)

    # Guarantee: If some classes have 0 samples in train, remove those classes and remap
    present = np.unique(y_tr)
    if len(present) < len(kept_ids):
        keep_present = np.array(sorted(present), dtype=int)
        remap = {old:i for i, old in enumerate(keep_present)}
        # Filter to keep only present classes
        tr_mask = np.isin(y_tr, keep_present)
        te_mask = np.isin(y_te, keep_present)
        subset = {
            "time_train": tr_raw["time"][tr_mask],
            "stat_train": tr_raw["stat"][tr_mask],
            "y_train":    np.array([remap[v] for v in y_tr[tr_mask]], dtype=int),
            "time_test":  te_raw["time"][te_mask],
            "stat_test":  te_raw["stat"][te_mask],
            "y_test":     np.array([remap[v] for v in y_te[te_mask]], dtype=int),
            "classes":    [classes[k] for k in kept_ids[keep_present]]
        }
    else:
        subset = {
            "time_train": tr_raw["time"],
            "stat_train": tr_raw["stat"],
            "y_train":    y_tr,
            "time_test":  te_raw["time"],
            "stat_test":  te_raw["stat"],
            "y_test":     y_te,
            "classes":    [classes[k] for k in kept_ids]
        }
    return subset

# ---------------- Train one scope ----------------
def train_and_eval_scope(tag, data_all, out_root, epochs_base=150, epochs_ft=50):
    if tag.lower().startswith("top"):
        n_first = int(tag.lower().replace("top",""))
    elif tag.lower() == "all":
        n_first = None
    else:
        raise ValueError(f"Unknown scope: {tag}")

    sub = subset_first_n_classes(data_all, n_first)
    num_classes = len(sub["classes"])
    scope_dir = os.path.join(out_root, f"{tag}_{num_classes}cls")
    os.makedirs(scope_dir, exist_ok=True)

    # Shapes and encoding
    x_time_tr = sub["time_train"][..., np.newaxis]  # (N, L, 1)
    x_stat_tr = sub["stat_train"]
    y_tr      = sub["y_train"]
    x_time_te = sub["time_test"][..., np.newaxis]
    x_stat_te = sub["stat_test"]
    y_te      = sub["y_test"]

    # Class weights (aligned with remapped 0..C-1)
    classes_arr = np.arange(num_classes, dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_tr)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_arr, class_weights)}

    # -------- Base training --------
    model = build_base_model(sequence_length=x_time_tr.shape[1],
                             num_stat_features=x_stat_tr.shape[1],
                             num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    base_path = os.path.join(scope_dir, f"base_model_{tag}.keras")
    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=20, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, mode="max", min_lr=1e-6),
        LearningRateScheduler(lr_schedule),
        ModelCheckpoint(base_path, monitor="val_accuracy", save_best_only=True)
    ]

    model.fit([x_time_tr, x_stat_tr],
              to_categorical(y_tr, num_classes=num_classes),
              validation_data=([x_time_te, x_stat_te], to_categorical(y_te, num_classes=num_classes)),
              epochs=epochs_base, batch_size=64,
              class_weight=class_weight_dict,
              callbacks=cbs, verbose=1)

    # Base evaluation
    model.load_weights(base_path)
    logits_base = model.predict([x_time_te, x_stat_te], verbose=0)
    y_pred_base = logits_base.argmax(axis=1)
    probs_base  = tf.nn.softmax(logits_base, axis=1).numpy()
    base_metrics = {
        "acc": accuracy_score(y_te, y_pred_base),
        "precision_macro": precision_score(y_te, y_pred_base, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_te, y_pred_base, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_te, y_pred_base, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_te, y_pred_base, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_te, y_pred_base, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_te, y_pred_base, average="weighted", zero_division=0),
    }

    # -------- Fine-tune --------
    ft_model = load_model(base_path, compile=False)  # Registered, can be loaded directly
    for layer in ft_model.layers:
        name = layer.name.lower()
        if ("rnn" in name) or ("dense" in name):
            layer.trainable = True
        else:
            layer.trainable = False
    ft_model.compile(optimizer=Adam(learning_rate=1e-5),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=["accuracy"])

    ft_model.fit([x_time_tr, x_stat_tr],
                 to_categorical(y_tr, num_classes=num_classes),
                 validation_data=([x_time_te, x_stat_te], to_categorical(y_te, num_classes=num_classes)),
                 epochs=epochs_ft, batch_size=32, verbose=1)

    logits_ft = ft_model.predict([x_time_te, x_stat_te], verbose=0)
    y_pred_ft = logits_ft.argmax(axis=1)
    probs_ft  = tf.nn.softmax(logits_ft, axis=1).numpy()

    ft_metrics = {
        "acc": accuracy_score(y_te, y_pred_ft),
        "precision_macro": precision_score(y_te, y_pred_ft, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_te, y_pred_ft, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_te, y_pred_ft, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_te, y_pred_ft, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_te, y_pred_ft, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_te, y_pred_ft, average="weighted", zero_division=0),
    }

    # Save fine-tuned model
    ft_path = os.path.join(scope_dir, f"fine_tuned_model_BIDMAGLSTM_{tag}.keras")
    ft_model.save(ft_path)

    # Save predictions (fine-tuned)
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": y_pred_ft,
        "prob_top1": probs_ft.max(axis=1)
    }).to_csv(os.path.join(scope_dir, "predictions.csv"), index=False)

    # Save metrics.json (including base and fine-tuned)
    with open(os.path.join(scope_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scope": f"{tag}_{num_classes}cls",
            "num_classes": num_classes,
            "train_samples": int(x_time_tr.shape[0]),
            "test_samples": int(x_time_te.shape[0]),
            "sequence_len": int(x_time_tr.shape[1]),
            "stat_dim": int(x_stat_tr.shape[1]),
            "base": {k: float(v) for k, v in base_metrics.items()},
            "fine_tuned": {k: float(v) for k, v in ft_metrics.items()},
            "class_weight_used": class_weight_dict
        }, f, ensure_ascii=False, indent=2)

    # Return summary row + class name mapping
    summary = {
        "scope": f"{tag}_{num_classes}cls",
        "acc": ft_metrics["acc"],
        "precision_macro": ft_metrics["precision_macro"],
        "recall_macro": ft_metrics["recall_macro"],
        "f1_macro": ft_metrics["f1_macro"],
        "precision_weighted": ft_metrics["precision_weighted"],
        "recall_weighted": ft_metrics["recall_weighted"],
        "f1_weighted": ft_metrics["f1_weighted"],
        "acc_base": base_metrics["acc"],
        "f1_macro_base": base_metrics["f1_macro"],
        "train_samples": x_time_tr.shape[0],
        "test_samples": x_time_te.shape[0],
        "num_classes": num_classes,
        "sequence_len": x_time_tr.shape[1],
        "model_base_path": base_path,
        "model_ft_path": ft_path
    }
    classes_df = pd.DataFrame({"class_id_new": np.arange(num_classes),
                               "class_name": sub["classes"]})
    return summary, classes_df

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="interval_time_closed_world_449.npz", help=".npz path")
    ap.add_argument("--outdir", default="bidmaglstm_runs", help="output root directory")
    ap.add_argument("--scopes", default="top100,top200", help="top100,top200,all (comma separated)")
    ap.add_argument("--epochs_base", type=int, default=150)
    ap.add_argument("--epochs_ft", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data_all = load_data(args.data)

    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    summary_rows, detail_sheets = [], {}

    for tag in scopes:
        row, df_detail = train_and_eval_scope(tag, data_all, args.outdir,
                                              epochs_base=args.epochs_base, epochs_ft=args.epochs_ft)
        summary_rows.append(row)
        detail_sheets[f"classes_{tag}"] = df_detail

    # Write Excel summary
    xlsx_path = os.path.join(args.outdir, "bidmaglstm_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for name, df in detail_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"\n[OK] Output directory: {args.outdir}")
    print(f"[OK] Summary Excel: {xlsx_path}")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n=== Summary (fine-tuned) ===")
        print(df[["scope","acc","f1_macro","train_samples","test_samples","num_classes"]])

if __name__ == "__main__":
    main()

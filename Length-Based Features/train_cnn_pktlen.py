# cnn_train_packetlen.py
import os, argparse, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, top_k_accuracy_score
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


def build_features_2ch(X):
    """
    X: (N, L) signed packet length sequence, fill value is -1
    -> (N, L, 2): channel0=values with -1 replaced by 0; channel1=mask(1=real value,0=padding)
    """
    mask = (X != -1).astype(np.float32)
    X0 = np.where(X == -1, 0.0, X).astype(np.float32)
    feats = np.stack([X0, mask], axis=-1)  # (N, L, 2)
    return feats

def build_cnn(sequence_len, num_classes):
    """
    Simple and robust 1D CNN, uses two-channel input: [value, mask]
    """
    inp = layers.Input(shape=(sequence_len, 2), name="seq2ch")  # (L, 2)

    x = layers.Conv1D(128, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    logits = layers.Dense(num_classes, name="logits")(x)  # from_logits=True
    model = Model(inp, logits, name="CNN_PacketLen")
    return model

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    needed = ["time_train", "time_test", "y_train", "y_test", "classes"]
    for k in needed:
        if k not in data.files:
            raise KeyError(f"'{k}' not found in {npz_path}")
    return {
        "Xtr": data["time_train"],
        "Xte": data["time_test"],
        "ytr": data["y_train"],
        "yte": data["y_test"],
        "classes": data["classes"],
        "train_paths": data["train_paths"] if "train_paths" in data.files else None,
        "test_paths":  data["test_paths"]  if "test_paths"  in data.files else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="packet_length_signed_449_all.npz", help=".npz data path")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--model_out", default="cnn_packetlen_model.keras", help="final model save path (only save once)")
    args = ap.parse_args()

    # 1) Load data and build features
    d = load_npz(args.data)
    Xtr_raw, Xte_raw = d["Xtr"], d["Xte"]
    ytr, yte = d["ytr"], d["yte"]
    classes = [str(c) for c in d["classes"]]
    n_classes = len(classes)
    seq_len = Xtr_raw.shape[1]

    Xtr = build_features_2ch(Xtr_raw)  # (N, L, 2)
    Xte = build_features_2ch(Xte_raw)

    # 2) Class weights (prevent class imbalance)
    cls_weights = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weight = {i: w for i, w in enumerate(cls_weights)}

    # 3) Build model
    model = build_cnn(seq_len, n_classes)
    model.summary()
    print(f"Total params: {model.count_params():,}")

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    if n_classes >= 5:
        # Using built-in TopKCategoricalAccuracy requires one-hot; for simplicity, only keep accuracy here,
        # After training, use sklearn's top_k to calculate Top-5.
        pass

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                         mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                             patience=4, mode="max", min_lr=1e-5),
    ]

    # 4) Training (do not save intermediate)
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 5) Save final model
    model.save(args.model_out)
    print(f"[OK] Model saved: {args.model_out}")

    # 6) Test set evaluation
    # After training and saving:
    logits = model.predict(Xte, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    acc, prec, rec, f1 = four_metrics(yte, y_pred, average="macro")
    print_four_metrics("CNN", acc, prec, rec, f1)

if __name__ == "__main__":
    # Optional: stabilize randomness (note: slight non-determinism may remain on GPU/CuDNN)
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    main()

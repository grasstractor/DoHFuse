# lstm_train_packetlen.py
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, top_k_accuracy_score
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

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    need = ["time_train", "time_test", "y_train", "y_test", "classes"]
    for k in need:
        if k not in data.files:
            raise KeyError(f"Missing '{k}' in {npz_path}")
    return (data["time_train"], data["time_test"],
            data["y_train"], data["y_test"], data["classes"])

def build_lstm(sequence_len, num_classes, units=128, bidirectional=True, stacked=True, dropout=0.3):
    """LSTM model:
       - Masking(-1) → (Bi)LSTM → (optional stack another layer) → fully connected classification head (logits)
    """
    inp = layers.Input(shape=(sequence_len, 1), name="seq")
    x = layers.Masking(mask_value=-1.0, name="mask")(inp)

    def rnn(units, return_sequences):
        core = layers.LSTM(units, return_sequences=return_sequences)
        return layers.Bidirectional(core, merge_mode="concat") if bidirectional else core

    # First layer
    x = rnn(units, return_sequences=stacked)(x)
    x = layers.Dropout(dropout)(x)

    # Second layer (optional)
    if stacked:
        x = rnn(units, return_sequences=False)(x)
        x = layers.Dropout(dropout)(x)
    # If not stacked, previous layer return_sequences=False is enough

    # Classification head (logits)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(num_classes, name="logits")(x)  # from_logits=True

    return Model(inp, logits, name="LSTM_PacketLen")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="packet_length_signed_449_all.npz", help=".npz data path")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--units",  type=int, default=128)
    ap.add_argument("--no_bidir", action="store_true", help="do not use bidirectional (default enabled)")
    ap.add_argument("--single_layer", action="store_true", help="use only one LSTM layer (default stacked two layers)")
    ap.add_argument("--model_out", default="lstm_packetlen_model.keras", help="final model save path (only save once)")
    args = ap.parse_args()

    # Stabilize randomness (CuDNN/LSTM may still have slight non-determinism)
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 1) Load data
    Xtr_raw, Xte_raw, ytr, yte, classes = load_npz(args.data)
    seq_len = Xtr_raw.shape[1]
    n_classes = len(classes)

    # 2) Shape adjustment: (N, L) -> (N, L, 1); Masking uses -1 as sentinel value
    Xtr = Xtr_raw[..., np.newaxis].astype(np.float32)
    Xte = Xte_raw[..., np.newaxis].astype(np.float32)

    # 3) Class weights
    cls_w = compute_class_weight(class_weight="balanced", classes=np.unique(ytr), y=ytr)
    class_weight = {i: w for i, w in enumerate(cls_w)}

    # 4) Build model
    model = build_lstm(
        sequence_len=seq_len,
        num_classes=n_classes,
        units=args.units,
        bidirectional=(not args.no_bidir),
        stacked=(not args.single_layer),
        dropout=0.3
    )
    model.summary()
    print(f"Total params: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                         mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                             patience=5, mode="max", min_lr=1e-5),
    ]

    # 5) Training (do not save intermediate)
    model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 6) Only save final model
    model.save(args.model_out)
    print(f"[OK] Model saved: {args.model_out}")

    # 7) Test set evaluation
    logits = model.predict(Xte, verbose=0)
    y_pred = np.argmax(logits, axis=1)
    acc, prec, rec, f1 = four_metrics(yte, y_pred, average="macro")
    print_four_metrics("LSTM", acc, prec, rec, f1)
    
    # acc = accuracy_score(yte, y_pred)
    # f1m = f1_score(yte, y_pred, average="macro")
    # f1w = f1_score(yte, y_pred, average="weighted")
    # print(f"[LSTM] Test Acc={acc:.4f} | Macro-F1={f1m:.4f} | Weighted-F1={f1w:.4f}")

    # if n_classes >= 5:
    #     probs = tf.nn.softmax(logits, axis=-1).numpy()
    #     top5 = top_k_accuracy_score(yte, probs, k=5, labels=np.arange(n_classes))
    #     print(f"[LSTM] Top-5 Acc={top5:.4f}")

    # # 8) Classification report
    # print("\n=== Classification Report ===")
    # print(classification_report(yte, y_pred, target_names=[str(c) for c in classes], digits=4))

if __name__ == "__main__":
    main()

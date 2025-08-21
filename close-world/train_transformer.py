# transformer_timeseries_scopes.py
import os, json, time, argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ========== Fix random seed ==========
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ========== Hyperparameters ==========
WORD_SIZE   = 1
SEQ_LEN     = 200
EMB_SIZE    = 64
NUM_HEADS   = 8
NUM_ENCODERS= 2
DENSE1      = 256
DENSE2      = 128
EPOCHS      = 150
BATCH_SIZE  = 128
LR          = 0.001
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Model definition ==========
class NetPatchEmbedding(nn.Module):
    def __init__(self, word_size, emb_size):
        super().__init__()
        self.linear = nn.Linear(word_size, emb_size)

    def forward(self, x):
        # x: (B, L, 1)
        x = self.linear(x)  # (B, L, E)
        # prepend CLS token (zeros)
        cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # (B, L+1, E)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, word_size, emb_size, num_classes):
        super().__init__()
        self.embedding = NetPatchEmbedding(word_size, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=NUM_HEADS, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODERS)
        self.norm = nn.LayerNorm(emb_size)
        self.fc1 = nn.Linear(emb_size, DENSE1)
        self.fc2 = nn.Linear(DENSE1, DENSE2)
        self.fc3 = nn.Linear(DENSE2, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        x: (B, L, 1), where padding value is -1
        Here we construct key_padding_mask: True means to mask.
        """
        # Mask first (before adding CLS)
        pad_mask = (x.squeeze(-1) == -1)  # (B, L) bool
        x = self.embedding(x)              # (B, L+1, E)

        # CLS is not masked; add a column of False at the front
        cls_mask = torch.zeros(pad_mask.size(0), 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # (B, L+1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L+1, E)
        x = self.norm(x[:, 0, :])  # Take CLS
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        logits = self.fc3(x)       # (B, C)
        return logits

# ========== Utility functions ==========
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    for k in ["time_train","time_test","y_train","y_test","classes"]:
        if k not in data.files:
            raise KeyError(f"Missing '{k}' in {npz_path}")
    Xtr = data["time_train"]  # (Ntr, L)
    Xte = data["time_test"]   # (Nte, L)
    ytr = data["y_train"].astype(int)
    yte = data["y_test"].astype(int)
    classes = [str(c) for c in data["classes"]]
    return Xtr, Xte, ytr, yte, classes

def subset_first_n_classes(Xtr, Xte, ytr, yte, classes, n_first=None):
    # Take the first n_first classes in order of classes; otherwise use all
    if (n_first is None) or (n_first >= len(classes)):
        kept = np.arange(len(classes))
    else:
        kept = np.array(list(range(n_first)), dtype=int)

    # Filter samples belonging to kept
    tr_mask = np.isin(ytr, kept)
    te_mask = np.isin(yte, kept)
    Xtr_s, Xte_s = Xtr[tr_mask], Xte[te_mask]
    ytr_s, yte_s = ytr[tr_mask], yte[te_mask]

    # Old id -> New id (0..C'-1)
    id_map = {old:i for i, old in enumerate(kept)}
    ytr_s = np.array([id_map[v] for v in ytr_s], dtype=int)
    yte_s = np.array([id_map[v] for v in yte_s], dtype=int)
    classes_s = [classes[old] for old in kept]
    return Xtr_s, Xte_s, ytr_s, yte_s, classes_s

def make_loader(X, y, batch_size, shuffle):
    # Keep -1 as padding value (used for mask in model), and expand to (B,L,1)
    x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

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

def train_one_scope(tag, Xtr0, Xte0, ytr0, yte0, classes0, out_root):
    """
    Train, evaluate, and save for one scope (e.g. top100 / top200 / all)
    """
    # Parse n_first
    if tag.lower().startswith("top"):
        n_first = int(tag.lower().replace("top",""))
    elif tag.lower() == "all":
        n_first = None
    else:
        raise ValueError(f"Unknown scope {tag}")

    # Subset
    Xtr, Xte, ytr, yte, classes = subset_first_n_classes(Xtr0, Xte0, ytr0, yte0, classes0, n_first)
    num_classes = len(classes)
    scope_dir = os.path.join(out_root, f"{tag}_{num_classes}cls")
    ensure_dir(scope_dir)

    # DataLoader
    train_loader = make_loader(Xtr, ytr, BATCH_SIZE, shuffle=True)
    test_loader  = make_loader(Xte, yte, BATCH_SIZE, shuffle=False)

    # Model
    model = TransformerClassifier(SEQ_LEN, WORD_SIZE, EMB_SIZE, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[INFO] === Scope: {tag} ({num_classes} classes) ===")
    print(f"[INFO] Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")

    # Training
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct = 0.0, 0
        t0 = time.time()
        for xb, yb in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_acc = correct / len(train_loader.dataset)
        print(f"[TRAIN][{tag}] epoch {epoch}  loss={total_loss/len(train_loader.dataset):.4f}  acc={tr_acc*100:.2f}%  time={time.time()-t0:.1f}s")

        # Simple test
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
        te_acc = correct / len(test_loader.dataset)
        print(f"[TEST ][{tag}] epoch {epoch}  acc={te_acc*100:.2f}%")

    # Final evaluation & save
    model.eval()
    all_logits, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_true.append(yb)
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = logits.argmax(1).numpy()
    probs  = torch.softmax(logits, dim=1).numpy()
    prob_top1 = probs.max(axis=1)

    metrics = eval_metrics(y_true, y_pred)

    # Save model
    model_path = os.path.join(scope_dir, f"transformer_closeworld_{num_classes}cls.pth")
    torch.save(model.state_dict(), model_path)

    # Save prediction details
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_top1": prob_top1
    }).to_csv(os.path.join(scope_dir, "predictions.csv"), index=False)

    # Save metrics JSON
    with open(os.path.join(scope_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scope": f"{tag}_{num_classes}cls",
            "num_classes": num_classes,
            "train_samples": int(len(train_loader.dataset)),
            "test_samples": int(len(test_loader.dataset)),
            "sequence_len": int(SEQ_LEN),
            "epochs": int(EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            **{k: float(v) for k, v in metrics.items()}
        }, f, ensure_ascii=False, indent=2)

    # Return summary row + class name table
    summary = {
        "scope": f"{tag}_{num_classes}cls",
        "acc": metrics["acc"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "num_classes": num_classes,
        "sequence_len": SEQ_LEN,
        "model_path": model_path
    }
    classes_df = pd.DataFrame({"class_id_new": np.arange(num_classes), "class_name": classes})
    return summary, classes_df, scope_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="interval_time_closed_world_449.npz", help=".npz data file")
    ap.add_argument("--outdir", default="transformer_runs", help="output root directory")
    ap.add_argument("--scopes", default="top100,top200,all",
                    help="evaluation scopes: top100,top200,all (comma separated, choose one)")
    args = ap.parse_args()

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading dataset from {args.data} ...")
    Xtr0, Xte0, ytr0, yte0, classes0 = load_npz(args.data)

    ensure_dir(args.outdir)

    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    summary_rows = []
    detail_sheets = {}

    for tag in scopes:
        summary, classes_df, scope_dir = train_one_scope(tag, Xtr0, Xte0, ytr0, yte0, classes0, args.outdir)
        summary_rows.append(summary)
        detail_sheets[f"classes_{tag}"] = classes_df

    # Write Excel summary
    xlsx_path = os.path.join(args.outdir, "transformer_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for name, df in detail_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"\n[OK] Output directory: {args.outdir}")
    print(f"[OK] Summary Excel: {xlsx_path}")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n=== Summary ===")
        cols = ["scope","acc","f1_macro","train_samples","test_samples","num_classes"]
        print(df[cols])

if __name__ == "__main__":
    main()

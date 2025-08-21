# train_fcnn_gru_ensemble_scopes.py
import os, json, copy, argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ================= Early Stopping =================
class EarlyStopping:
    """
    Monitor val_macro_f1; stop when no improvement for consecutive 'patience' epochs.
    Save best state_dict and restore after training.
    """
    def __init__(self, patience=5, min_delta=1e-6, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.best_state = None

    def step(self, metric, model):
        improved = False
        if self.best is None:
            improved = True
        else:
            improved = (metric - self.best) > self.min_delta if self.mode=="max" else (self.best - metric) > self.min_delta
        if improved:
            self.best = metric
            self.num_bad = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ================= Data Loading/Subset =================
def load_npz_all(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    need = ["time_q_train","time_q_test","time_r_train","time_r_test",
            "q_valid_len_train","q_valid_len_test","r_valid_len_train","r_valid_len_test",
            "fcnn_train","fcnn_test","y_train","y_test","classes"]
    for k in need:
        if k not in d.files:
            raise KeyError(f"Missing '{k}' in {npz_path}")
    return {k: d[k] for k in need}

def subset_first_n(data, n_first=None):
    """
    Take the first n_first classes from 'classes'; compress y to 0..C'-1.
    If a class has 0 samples in the training set, remove and remap.
    """
    classes_all = list(map(str, data["classes"]))
    if (n_first is None) or (n_first >= len(classes_all)):
        kept_old = np.arange(len(classes_all), dtype=int)
    else:
        kept_old = np.arange(n_first, dtype=int)

    def _split(name):
        y_old = data[f"y_{name}"].astype(int)
        mask  = np.isin(y_old, kept_old)
        id_map = {old:i for i, old in enumerate(kept_old)}
        y_new = np.array([id_map[v] for v in y_old[mask]], dtype=int)
        out = {
            "tq": data[f"time_q_{name}"][mask].astype(np.float32),
            "tr": data[f"time_r_{name}"][mask].astype(np.float32),
            "ql": data[f"q_valid_len_{name}"][mask].astype(np.int32),
            "rl": data[f"r_valid_len_{name}"][mask].astype(np.int32),
            "f6": data[f"fcnn_{name}"][mask].astype(np.float32),
            "y":  y_new
        }
        return out

    tr = _split("train")
    te = _split("test")

    present = np.unique(tr["y"])
    if len(present) < len(np.unique(tr["y"])) or len(np.unique(tr["y"])) < len(kept_old):
        # Guarantee: keep only classes present in training set
        keep = np.array(sorted(np.unique(tr["y"])), dtype=int)
        remap = {old:i for i, old in enumerate(keep)}
        def _filter_pack(pack):
            m = np.isin(pack["y"], keep)
            return {
                "tq": pack["tq"][m], "tr": pack["tr"][m],
                "ql": pack["ql"][m], "rl": pack["rl"][m],
                "f6": pack["f6"][m], "y": np.array([remap[v] for v in pack["y"][m]], dtype=int)
            }
        tr = _filter_pack(tr)
        te = _filter_pack(te)
        classes_new = [classes_all[kept_old[i]] for i in keep]
    else:
        classes_new = [classes_all[i] for i in kept_old]
    return tr, te, classes_new

# ================= Dataset & Collate =================
class ODoHArraysDataset(Dataset):
    """
    Construct directly from arrays:
      - tq,tr: (N, L) padding=-1
      - ql,rl: (N,)
      - f6: (N,6) -> 6th dim TT at idx=5
      - y: (N,)
    """
    def __init__(self, tq, tr, ql, rl, f6, y, normalize_timestamps=True):
        self.tq, self.tr, self.ql, self.rl, self.f6, self.y = tq, tr, ql, rl, f6, y
        self.normalize = normalize_timestamps
        self.TT_index = 5
        n = len(self.y)
        assert all(len(arr) == n for arr in [self.tq, self.tr, self.ql, self.rl, self.f6])

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        tq = self.tq[idx].copy(); tr = self.tr[idx].copy()
        ql_raw = int(self.ql[idx]); rl_raw = int(self.rl[idx])
        f6 = self.f6[idx].copy();   y = int(self.y[idx])
        L = int(tq.shape[0])
        ql = min(max(ql_raw,0), L); rl = min(max(rl_raw,0), L)
        if self.normalize:
            t0 = float(tq[0]) if ql>0 else (float(tr[0]) if rl>0 else 0.0)
            TT = float(f6[self.TT_index]); denom = TT if TT>1e-6 else 1.0
            if ql>0: tq[:ql] = (tq[:ql] - t0) / denom
            if rl>0: tr[:rl] = (tr[:rl] - t0) / denom
        eff_len = max(ql, rl) if max(ql, rl)>0 else 1
        seq = np.stack([tq, tr], axis=-1).astype(np.float32)
        return torch.from_numpy(seq), torch.tensor(eff_len), torch.from_numpy(f6).float(), torch.tensor(y).long()

def collate_pad(batch):
    seqs, lens, f6s, ys = zip(*batch)
    clamped, Lmax = [], 1
    for s, l in zip(seqs, lens):
        l = int(l.item()); l = min(l, s.shape[0]); clamped.append(l); Lmax = max(Lmax, l)
    B = len(batch)
    out = torch.full((B, Lmax, 2), -1.0, dtype=torch.float32)
    for i, (s, l) in enumerate(zip(seqs, clamped)): out[i, :l, :] = s[:l, :]
    return out, torch.tensor(clamped, dtype=torch.int64), torch.stack(f6s).float(), torch.stack(ys).long()

# ================= Models (Ensemble Soft-Vote) =================
class FCNNClassifier(nn.Module):
    def __init__(self, in_dim=6, num_classes=10):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),    nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.cls = nn.Linear(64, num_classes)
    def forward(self, x): return self.cls(self.feat(x))

class GRUClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=1028, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256),         nn.ReLU(),
            nn.Linear(256, 128),         nn.ReLU(),
            nn.Linear(128, 128),         nn.ReLU(),
        )
        self.cls = nn.Linear(128, num_classes)
    def forward(self, seq, lengths):
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)      # h: [1,B,H]
        f = self.proj(h[-1])         # [B,128]
        return self.cls(f)

class EnsembleFCNNGRU(nn.Module):
    """
    Two-branch logits -> soft-vote probability average (or learnable weights), output ensemble logits (log(prob)).
    """
    def __init__(self, num_classes, fcnn_in_dim=6, learn_weights=False):
        super().__init__()
        self.fcnn = FCNNClassifier(in_dim=fcnn_in_dim, num_classes=num_classes)
        self.gru  = GRUClassifier(input_size=2, hidden_size=1028, num_classes=num_classes)
        self.learn_weights = learn_weights
        if learn_weights:
            self.alpha = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
    def forward(self, f6, seq, lengths):
        lf = self.fcnn(f6)
        lg = self.gru(seq, lengths)
        if self.learn_weights:
            w = torch.softmax(self.alpha, dim=0)
            prob = w[0]*torch.softmax(lf, dim=1) + w[1]*torch.softmax(lg, dim=1)
        else:
            prob = 0.5*(torch.softmax(lf, dim=1) + torch.softmax(lg, dim=1))
        return lf, lg, torch.log(prob + 1e-8)  # Return (branch logits, ensemble logits)

# ================= Evaluation (using ensemble output) =================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    ce = nn.CrossEntropyLoss()
    loss_sum, n_sum = 0.0, 0
    with torch.no_grad():
        for seq, lend, f6, y in loader:
            seq, lend, f6, y = seq.to(device), lend.to(device), f6.to(device), y.to(device)
            lf, lg, lens = model(f6, seq, lend)
            # Validation loss: average CE of two branches
            loss = 0.5*(ce(lf, y) + ce(lg, y))
            loss_sum += float(loss.item()) * y.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(lens.argmax(1).cpu().numpy())
            n_sum += y.size(0)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return (loss_sum / max(1,n_sum)), acc, p_m, r_m, f1_m

def predict_probs(model, loader, device):
    """Return y_true, y_pred, prob_top1 (based on ensemble output)"""
    model.eval()
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for seq, lend, f6, y in loader:
            seq, lend, f6, y = seq.to(device), lend.to(device), f6.to(device), y.to(device)
            _, _, lens = model(f6, seq, lend)
            p = torch.softmax(lens, dim=1)
            probs.append(p.cpu().numpy())
            y_true.append(y.cpu().numpy())
            y_pred.append(p.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred); probs = np.concatenate(probs, axis=0)
    return y_true, y_pred, probs.max(axis=1)

# ================= Single scope: training + save only final best =================
def train_one_scope(tag, data_all, out_root, batch_size=64, lr=1e-3, max_epochs=100,
                    patience=5, learn_weights=False):
    # Parse scope
    if tag.lower().startswith("top"):
        n_first = int(tag.lower().replace("top",""))
    elif tag.lower() == "all":
        n_first = None
    else:
        raise ValueError(f"Unknown scope: {tag}")

    # Subset & class names
    tr, te, classes = subset_first_n(data_all, n_first)
    scope_dir = os.path.join(out_root, f"{tag}_{len(classes)}cls")
    os.makedirs(scope_dir, exist_ok=True)

    # Dataset / Loader
    train_ds = ODoHArraysDataset(tr["tq"], tr["tr"], tr["ql"], tr["rl"], tr["f6"], tr["y"], normalize_timestamps=True)
    test_ds  = ODoHArraysDataset(te["tq"], te["tr"], te["ql"], te["rl"], te["f6"], te["y"], normalize_timestamps=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleFCNNGRU(num_classes=len(classes), fcnn_in_dim=tr["f6"].shape[1], learn_weights=learn_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience, mode="max")

    print(f"[{tag}] classes={len(classes)} | train={len(train_ds)} | test={len(test_ds)} | device={device}")
    # Training
    for ep in range(1, max_epochs+1):
        model.train()
        loss_sum, n_sum = 0.0, 0
        for seq, lend, f6, y in train_loader:
            seq, lend, f6, y = seq.to(device), lend.to(device), f6.to(device), y.to(device)
            optimizer.zero_grad()
            lf, lg, _ = model(f6, seq, lend)
            loss = 0.5*(ce(lf, y) + ce(lg, y))
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * y.size(0); n_sum += y.size(0)
        tr_loss = loss_sum / max(1, n_sum)

        ev_loss, ev_acc, ev_p, ev_r, ev_f1 = evaluate(model, test_loader, device)
        print(f"[{tag}] EP{ep:03d} | train_loss={tr_loss:.4f} | val_loss={ev_loss:.4f} "
              f"| acc={ev_acc:.4f} | Pm={ev_p:.4f} | Rm={ev_r:.4f} | F1m={ev_f1:.4f}")

        if stopper.step(ev_f1, model):
            print(f"[{tag}] Early stopping at epoch {ep}, best macro-F1={stopper.best:.4f}")
            break

    # Restore best & final evaluation
    stopper.restore(model)
    ev_loss, ev_acc, ev_p, ev_r, ev_f1 = evaluate(model, test_loader, device)

    # Prediction details
    y_true, y_pred, prob_top1 = predict_probs(model, test_loader, device)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "prob_top1": prob_top1}).to_csv(
        os.path.join(scope_dir, "predictions.csv"), index=False
    )

    # Save only final best model
    model_path = os.path.join(scope_dir, f"fcnn_gru_ensemble_{tag}.pth")
    torch.save(model.state_dict(), model_path)

    # metrics.json
    metrics = {
        "scope": f"{tag}_{len(classes)}cls",
        "num_classes": len(classes),
        "train_samples": int(len(train_ds)),
        "test_samples": int(len(test_ds)),
        "sequence_len": int(tr["tq"].shape[1]),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "learn_weights": bool(learn_weights),
        "acc": float(ev_acc),
        "precision_macro": float(ev_p),
        "recall_macro": float(ev_r),
        "f1_macro": float(ev_f1),
        "model_path": model_path
    }
    with open(os.path.join(scope_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Class name table
    classes_df = pd.DataFrame({"class_id_new": np.arange(len(classes)), "class_name": classes})

    # Summary row
    summary = {
        "scope": f"{tag}_{len(classes)}cls",
        "acc": ev_acc, "precision_macro": ev_p, "recall_macro": ev_r, "f1_macro": ev_f1,
        "train_samples": len(train_ds), "test_samples": len(test_ds),
        "num_classes": len(classes), "sequence_len": int(tr["tq"].shape[1]),
        "model_path": model_path
    }
    return summary, classes_df, scope_dir

# ================= Main: three scopes + Excel =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="odoh_e3_fcnn_gru_closed_world.npz", help=".npz data file")
    ap.add_argument("--outdir", default="fcnn_gru_ensemble_runs", help="output root directory")
    ap.add_argument("--scopes", default="top100,top200,all", help="evaluation scopes, comma separated (top100,top200,all)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--learn_weights", action="store_true", help="use learnable fusion weights")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data_all = load_npz_all(args.data)

    summaries, detail_sheets = [], {}
    for tag in [s.strip() for s in args.scopes.split(",") if s.strip()]:
        summary, classes_df, _ = train_one_scope(tag, data_all, args.outdir,
                                                 batch_size=args.batch_size,
                                                 lr=args.lr,
                                                 max_epochs=args.max_epochs,
                                                 patience=args.patience,
                                                 learn_weights=args.learn_weights)
        summaries.append(summary)
        detail_sheets[f"classes_{tag}"] = classes_df

    # Excel summary
    xlsx_path = os.path.join(args.outdir, "fcnn_gru_ensemble_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="summary", index=False)
        for name, df in detail_sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"\n[OK] Output directory: {args.outdir}")
    print(f"[OK] Summary Excel: {xlsx_path}")
    if summaries:
        df = pd.DataFrame(summaries)
        print("\n=== Summary ===")
        print(df[["scope","acc","f1_macro","train_samples","test_samples","num_classes"]])

if __name__ == "__main__":
    main()

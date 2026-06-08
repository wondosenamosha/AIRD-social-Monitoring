"""
Retrain the Custom Transformer (Track B) as a self-contained serving model and
save its weights TOGETHER WITH the word2idx vocab (the original .pt shipped a
bare state_dict with no vocab/tokenizer, so it could not be served).

Faithful to notebooks/AIRD_Project_Notebook.ipynb (cells 24, 28-30):
  vocab 30000 (<PAD>,<OOV>+most_common from X_tr_o) · GloVe-300d init ·
  4 transformer blocks · 6 heads · ff 1200 · mean-pool -> fc1(300->150)->fc2(7) ·
  FocalLoss(gamma2, ls0.05) · AdamW + warmup/cosine · GloVe emb lr/5 · clip 0.5.

Runs on Apple MPS if available (CUDA autocast/scaler replaced with plain fp32).

    python scripts/train_transformer.py --epochs 45
Writes webapp/models/CustomTransformer_TrackB/{state_dict.pt,word2idx.json,config.json}
"""
import os, sys, json, time, math, argparse
from collections import Counter
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
LABELS = ["Anxiety", "Bipolar", "Depression", "Normal",
          "Personality Disorder", "Stress", "Suicidal"]
LABEL_MAP = {c: i for i, c in enumerate(LABELS)}
NUM_CLASSES = 7
MAX_LEN = 256
VOCAB_SIZE = 30000

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.dirname(HERE)
sys.path.insert(0, WEBAPP)
from ct_model import CustomTransformer  # shared architecture (matches inference.py)

DEFAULT_DATA = "/Users/macbbok/Downloads/Takeout/Drive/AIRD_Project 3/data"
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__(); self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        n = logits.size(-1)
        logp = F.log_softmax(logits.float(), dim=-1)
        pt = logp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        sm = self.ls / max(n - 1, 1)
        ce = -(sm * logp.sum(-1) + (1.0 - self.ls - sm) * logp.gather(1, targets.unsqueeze(1)).squeeze(1))
        return (((1 - pt) ** self.gamma) * ce).mean()


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long); self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def texts_to_padded(texts, w2i, max_len=MAX_LEN):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [w2i.get(w, 1) for w in t.split()][:max_len]
        out[i, :len(ids)] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(WEBAPP, "models", "CustomTransformer_TrackB"))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {DEVICE}")

    df = pd.read_csv(os.path.join(args.data, "Combined_Data_cleaned.csv")).dropna(
        subset=["statement", "status"]).reset_index(drop=True)
    X, y = df["statement"].astype(str).values, df["status"].values
    X_tr_o, X_tmp, y_tr_o, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    bal = pd.read_csv(os.path.join(args.data, "Combined_Data_balanced.csv")).dropna(subset=["statement", "status"])
    aug = bal[~bal["statement"].astype(str).isin(set(df["statement"].astype(str)))]
    X_tr_b = np.concatenate([X_tr_o, aug["statement"].astype(str).values])
    y_tr_b = np.concatenate([y_tr_o, aug["status"].values])
    print(f"train_b={len(X_tr_b):,} val={len(X_val):,} test={len(X_test):,}")

    # vocab from original train
    freq = Counter()
    for t in X_tr_o:
        freq.update(t.split())
    vocab = ["<PAD>", "<OOV>"] + [w for w, _ in freq.most_common(VOCAB_SIZE - 2)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    print(f"vocab={len(vocab):,}")

    # GloVe init
    glove = np.zeros((len(vocab), 300), dtype=np.float32)
    gpath = os.path.join(args.data, "glove.6B.300d.txt")
    hit = 0
    with open(gpath, encoding="utf-8") as gf:
        for line in gf:
            p = line.rstrip().split(" ")
            if p[0] in word2idx:
                glove[word2idx[p[0]]] = np.asarray(p[1:], dtype=np.float32); hit += 1
    glove[0] = 0.0
    print(f"GloVe coverage {hit:,}/{len(vocab)-2:,} ({100*hit/(len(vocab)-2):.1f}%)")

    Xtr = texts_to_padded(X_tr_b, word2idx); Xv = texts_to_padded(X_val, word2idx)
    Xte = texts_to_padded(X_test, word2idx)
    ytr = np.array([LABEL_MAP[l] for l in y_tr_b])
    yv = np.array([LABEL_MAP[l] for l in y_val]); yte = np.array([LABEL_MAP[l] for l in y_test])

    loader = DataLoader(SeqDS(Xtr, ytr), batch_size=args.batch, shuffle=True, num_workers=0)
    model = CustomTransformer(VOCAB_SIZE, MAX_LEN, NUM_CLASSES, glove_matrix=glove).to(DEVICE)

    emb_ids = {id(p) for p in model.tok_emb.parameters()}
    rest = [p for p in model.parameters() if id(p) not in emb_ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                             {"params": list(model.tok_emb.parameters()), "lr": args.lr / 5.0}],
                            weight_decay=0.01, eps=1e-8)
    total = args.epochs * len(loader); warm = max(1, int(0.10 * total)); minf = 1e-2
    def lr_lambda(s):
        if s < warm: return s / warm
        pr = (s - warm) / max(1, total - warm)
        return minf + 0.5 * (1 - minf) * (1 + math.cos(math.pi * pr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = FocalLoss(gamma=2.0, label_smoothing=0.05)

    Xv_t = torch.tensor(Xv, dtype=torch.long).to(DEVICE)
    best_f1, best_state, no_imp = 0.0, None, 0
    for ep in range(args.epochs):
        model.train(); tl, ns = 0.0, 0; t0 = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step()
            tl += loss.item(); ns += 1
        model.eval()
        with torch.no_grad():
            preds = model(Xv_t).argmax(-1).cpu().numpy()
        f1 = f1_score(yv, preds, average="macro", zero_division=0)
        print(f"  Ep {ep+1:3d}/{args.epochs}  train={tl/ns:.4f}  valF1={f1:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1 = f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"  early stop (best valF1={best_f1:.4f})"); break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tp = model(torch.tensor(Xte, dtype=torch.long).to(DEVICE)).argmax(-1).cpu().numpy()
    print(f"  [CustomTransformer-TrackB] acc={accuracy_score(yte,tp):.4f}  "
          f"macroF1={f1_score(yte,tp,average='macro'):.4f}  "
          f"balAcc={balanced_accuracy_score(yte,tp):.4f}  MCC={matthews_corrcoef(yte,tp):.4f}")
    print("  published: acc 0.758  macroF1 0.790")

    torch.save(best_state, os.path.join(args.out, "state_dict.pt"))
    json.dump(word2idx, open(os.path.join(args.out, "word2idx.json"), "w"))
    json.dump({"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN, "num_classes": NUM_CLASSES,
               "embed_dim": 300, "num_heads": 6, "ff_dim": 1200, "num_blocks": 4,
               "labels": LABELS}, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()

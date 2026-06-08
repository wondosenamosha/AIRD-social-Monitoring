"""
Retrain self-contained LogReg + LightGBM Track-B serving pipelines and save them
WITH their TF-IDF vectorizers (the originals shipped the classifier only, so the
saved weights can't be re-aligned to a re-fit vectorizer).

Faithful to notebooks/AIRD_Project_Notebook.ipynb (cells 23-25):
  * split  : stratified 70/15/15 of Combined_Data_cleaned.csv, seed 42
  * TF-IDF : word(35k,1-2g,min_df2) + char(20k,3-6,char_wb,min_df3), fit on X_tr_o
  * train  : on Track-B augmented set (orig train + augmented rows) = 89,256
Validated: retrained LogReg reproduces the published 0.745 acc / 0.760 macro-F1.

    python scripts/train_classical.py
Writes webapp/models/classical/{logreg,lgbm}_trackb.joblib  (word_vec,char_vec,clf,classes)
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.sparse import hstack as sp_hstack

SEED = 42
LABELS = ["Anxiety", "Bipolar", "Depression", "Normal",
          "Personality Disorder", "Stress", "Suicidal"]
LABEL_MAP = {c: i for i, c in enumerate(LABELS)}

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.dirname(HERE)
DEFAULT_DATA = "/Users/macbbok/Downloads/Takeout/Drive/AIRD_Project 3/data"


def load_splits(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "Combined_Data_cleaned.csv"))
    df = df.dropna(subset=["statement", "status"]).reset_index(drop=True)
    X, y = df["statement"].astype(str).values, df["status"].values
    X_tr_o, X_tmp, y_tr_o, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    # Track B augmented training = original train + augmented-only rows
    bal = pd.read_csv(os.path.join(data_dir, "Combined_Data_balanced.csv"))
    bal = bal.dropna(subset=["statement", "status"])
    orig = set(df["statement"].astype(str))
    aug = bal[~bal["statement"].astype(str).isin(orig)]
    X_tr_b = np.concatenate([X_tr_o, aug["statement"].astype(str).values])
    y_tr_b = np.concatenate([y_tr_o, aug["status"].values])
    return X_tr_o, X_tr_b, y_tr_b, X_val, y_val, X_test, y_test


def metrics(tag, y_true, y_pred):
    yt = np.array([LABEL_MAP[l] for l in y_true]) if y_true.dtype == object else y_true
    acc = accuracy_score(yt, y_pred)
    f1m = f1_score(yt, y_pred, average="macro", zero_division=0)
    bal = f1_score(yt, y_pred, average="macro", zero_division=0)  # placeholder
    from sklearn.metrics import balanced_accuracy_score
    bal = balanced_accuracy_score(yt, y_pred)
    mcc = matthews_corrcoef(yt, y_pred)
    print(f"  [{tag}] acc={acc:.4f}  macroF1={f1m:.4f}  balAcc={bal:.4f}  MCC={mcc:.4f}")
    return acc, f1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(WEBAPP, "models", "classical"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading + reproducing splits...")
    X_tr_o, X_tr_b, y_tr_b, X_val, y_val, X_test, y_test = load_splits(args.data)
    print(f"  train_orig={len(X_tr_o):,}  train_aug={len(X_tr_b):,}  "
          f"val={len(X_val):,}  test={len(X_test):,}")
    yb = np.array([LABEL_MAP[l] for l in y_tr_b])
    yv = np.array([LABEL_MAP[l] for l in y_val])
    yt = np.array([LABEL_MAP[l] for l in y_test])

    print("Fitting TF-IDF (word 35k + char 20k) on original train...")
    word_vec = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True,
                               min_df=2, strip_accents="unicode", analyzer="word").fit(X_tr_o)
    char_vec = TfidfVectorizer(max_features=20000, ngram_range=(3, 6), sublinear_tf=True,
                               min_df=3, analyzer="char_wb").fit(X_tr_o)

    def tf(texts):
        return sp_hstack([word_vec.transform(texts), char_vec.transform(texts)], format="csr")

    Xb, Xv, Xt = tf(X_tr_b), tf(X_val), tf(X_test)
    print(f"  features = {Xb.shape[1]:,}")

    # ---- Logistic Regression (Track B params) ---------------------------- #
    print("Training LogReg...")
    t0 = time.time()
    lr = LogisticRegression(C=0.7, penalty="l2", solver="lbfgs", max_iter=500,
                            tol=1e-4, class_weight=None, n_jobs=-1, random_state=SEED)
    lr.fit(Xb, yb)
    metrics("LogReg-TrackB", y_test, lr.predict(Xt))
    joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": lr, "classes": LABELS},
                os.path.join(args.out, "logreg_trackb.joblib"))
    print(f"  saved logreg_trackb.joblib  ({time.time()-t0:.0f}s)")

    # ---- LightGBM (Track B params) --------------------------------------- #
    print("Training LightGBM...")
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    t0 = time.time()
    sw = np.ones(len(yb), dtype=np.float32)
    sw[np.isin(yb, [LABEL_MAP["Suicidal"], LABEL_MAP["Depression"]])] = 1.5
    gbm = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=127,
                         max_depth=8, min_child_samples=15, colsample_bytree=0.7,
                         subsample=0.8, subsample_freq=1, reg_alpha=0.05, reg_lambda=0.1,
                         min_split_gain=0.005, path_smooth=1, class_weight=None,
                         random_state=SEED, n_jobs=-1, verbose=-1)
    gbm.fit(Xb, yb, sample_weight=sw, eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    metrics("LightGBM-TrackB", y_test, gbm.predict(Xt))
    joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": gbm, "classes": LABELS},
                os.path.join(args.out, "lgbm_trackb.joblib"))
    print(f"  saved lgbm_trackb.joblib  ({time.time()-t0/60:.1f}... {time.time()-t0:.0f}s)")
    print("DONE.")


if __name__ == "__main__":
    main()

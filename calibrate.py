"""
Temperature calibration for AIRD inference models.

Finds the T in softmax(logits/T) that minimises Negative Log-Likelihood
on a stratified sample of the cleaned dataset.

Usage (run where models are loaded — Railway or local with enough RAM):
    python3 calibrate.py [--n 10] [--model ensemble|mentalbert|logreg]

Paste the printed TEMPERATURE value into inference.py.
"""
import argparse, math, random, os, sys
import pandas as pd
from scipy.optimize import minimize_scalar

LABELS = ["Normal", "Stress", "Anxiety", "Personality Disorder",
          "Bipolar", "Depression", "Suicidal"]

CSV_DEFAULT = os.path.join(os.path.dirname(__file__),
    "../Downloads/Takeout/Drive/AIRD_Project 3/data/Combined_Data_cleaned.csv")


# ── temperature helpers ───────────────────────────────────────────────────────

def apply_temp(probs: dict, T: float) -> dict:
    log_p = {k: math.log(max(v, 1e-10)) / T for k, v in probs.items()}
    shift = max(log_p.values())
    exp_p = {k: math.exp(v - shift) for k, v in log_p.items()}
    Z     = sum(exp_p.values())
    return {k: v / Z for k, v in exp_p.items()}

def nll(rows, T):
    return -sum(math.log(max(apply_temp(r["probs"], T)[r["true"]], 1e-10))
                for r in rows) / len(rows)

def ece(rows, T, bins=10):
    bucket = [[] for _ in range(bins)]
    for r in rows:
        recal = apply_temp(r["probs"], T)
        top   = max(recal, key=recal.get)
        conf  = recal[top]
        bucket[min(int(conf * bins), bins - 1)].append(
            (conf, int(top == r["true"])))
    val = sum(len(b) * abs(sum(x[0] for x in b) / len(b) -
                           sum(x[1] for x in b) / len(b))
              for b in bucket if b)
    return val / len(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=10)
    ap.add_argument("--model", default="ensemble",
                    help="ensemble | mentalbert | roberta | logreg | lgbm")
    ap.add_argument("--csv",   default=CSV_DEFAULT)
    args = ap.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df.columns = ["statement", "label"]

    random.seed(42)
    sample = []
    for lbl in LABELS:
        rows = df[df["label"] == lbl]["statement"].tolist()
        random.shuffle(rows)
        sample += [(t, lbl) for t in rows[: args.n]]
    random.shuffle(sample)

    # ── load inference module ─────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))

    # macOS: set DYLD for PyTorch before importing torch
    try:
        import subprocess
        lib = subprocess.check_output(
            ["python3", "-c",
             "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"],
            text=True).strip()
        os.environ.setdefault("DYLD_LIBRARY_PATH", lib)
    except Exception:
        pass

    import inference

    # Temporarily disable temperature scaling so we get raw model probs
    _orig_T = inference.TEMPERATURE
    inference.TEMPERATURE = 1.0

    print(f"\nCalibrating on {len(sample)} examples ({args.n}/class) "
          f"with model='{args.model}'…\n")

    rows = []
    for i, (text, true_label) in enumerate(sample):
        print(f"  [{i+1:02d}/{len(sample)}] {true_label[:20]:20}", end=" ", flush=True)
        try:
            res = inference.analyze(text, args.model)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        probs = {r["emotion"]: r["pct"] / 100 for r in res["ranked"]}
        rows.append({"true": true_label, "probs": probs,
                     "predicted": res["top_emotion"],
                     "confidence": res["confidence"]})
        ok = "✅" if res["top_emotion"] == true_label else "❌"
        print(f"→ {res['top_emotion']:22} {res['confidence']:5.1f}%  {ok}")

    inference.TEMPERATURE = _orig_T   # restore

    if not rows:
        print("\nNo results — model failed to load.")
        return

    # ── optimise T ────────────────────────────────────────────────────────────
    acc    = sum(r["predicted"] == r["true"] for r in rows) / len(rows)
    result = minimize_scalar(lambda T: nll(rows, T),
                             bounds=(0.5, 1.5), method="bounded")
    T_opt  = round(result.x, 3)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CALIBRATION RESULTS  (model: {args.model})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Samples              : {len(rows)}
 Accuracy             : {acc:.1%}
 Optimal temperature  : T = {T_opt}

 NLL  T=1.00 (raw)    : {nll(rows, 1.0):.4f}
 NLL  T={T_opt}         : {nll(rows, T_opt):.4f}
 ECE  T=1.00 (raw)    : {ece(rows, 1.0):.4f}
 ECE  T={T_opt}         : {ece(rows, T_opt):.4f}

 ✅  Set TEMPERATURE = {T_opt} in inference.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()

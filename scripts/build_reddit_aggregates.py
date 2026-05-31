"""
Precompute the Awario-style dashboard aggregates from the EXISTING Reddit
inference output (results/reddit_inference_results.csv) into a single JSON the
web app serves. No model is loaded here — this only summarises predictions that
were already produced by the saved research models.

Run from the project root:
    python webapp/scripts/build_reddit_aggregates.py
"""
import os
import re
import sys
import json
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.dirname(HERE)
ROOT = os.path.dirname(WEBAPP)
sys.path.insert(0, WEBAPP)
from inference import EMOTION_META, LABELS  # noqa: E402

SRC = os.path.join(ROOT, "results", "reddit_inference_results.csv")
MASTER = os.path.join(ROOT, "results", "master_comparison_table.csv")
OUT = os.path.join(WEBAPP, "data", "reddit_dashboard.json")
MODEL_NOTE = "Ensemble (Track B) · 84.6% acc / 0.90 macro-F1"

# Which Track B models can actually serve live inference (complete artifacts).
LIVE_LOADABLE = {"MentalBERT", "RoBERTa"}
MODEL_DISPLAY = {
    "Ensemble": "Stacking Ensemble",
    "MentalBERT": "MentalBERT",
    "RoBERTa": "Mental-RoBERTa",
    "LightGBM": "LightGBM",
    "CustomTransformer": "Custom Transformer",
    "LogReg": "Logistic Regression",
}


def build_leaderboard():
    """Track B leaderboard from the published master comparison table."""
    if not os.path.exists(MASTER):
        return []
    m = pd.read_csv(MASTER)
    m = m[m["Track"] == "TrackB"].sort_values("F1_Macro", ascending=False)
    rows = []
    for _, r in m.iterrows():
        name = r["Model"]
        rows.append({
            "model": MODEL_DISPLAY.get(name, name),
            "raw": name,
            "accuracy": round(100 * float(r["Accuracy"]), 1),
            "f1_macro": round(100 * float(r["F1_Macro"]), 1),
            "f1_weighted": round(100 * float(r["F1_Weighted"]), 1),
            "live": name in LIVE_LOADABLE,
        })
    return rows

_TOKEN_RE = re.compile(r"[a-z]{3,}")
_EXTRA_STOP = {
    "like", "just", "people", "really", "going", "know", "think", "want",
    "would", "could", "even", "still", "make", "much", "way", "get", "got",
    "one", "see", "say", "said", "thing", "things", "time", "year", "years",
    "day", "work", "job", "company", "ai", "https", "http", "www", "com",
    "amp", "removed", "deleted", "edit", "lol", "yeah", "gonna", "dont",
    "don", "doesn", "didn", "isn", "wasn", "aren", "wouldn", "couldn",
    "shouldn", "won", "ain", "doing", "getting", "going", "good", "right",
}
STOP = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP

# Light display-only mask so the public dashboard does not lead with slurs.
_PROFANITY = re.compile(
    r"\b(fuck(?:ing|ers?|ed)?|shit(?:ty)?|bitch(?:es)?|cunts?|asshole?s?|"
    r"motherfuck\w*|dick(?:head)?s?)\b", re.IGNORECASE)


def mask_profanity(text: str) -> str:
    return _PROFANITY.sub(lambda m: m.group(0)[0] + "•" * (len(m.group(0)) - 1), text)


def snippet(text: str, n: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = mask_profanity(text)
    return text[:n] + ("…" if len(text) > n else "")


def main():
    df = pd.read_csv(SRC)
    df = df.dropna(subset=["text", "predicted_label"])
    df = df[df["predicted_label"].isin(LABELS)].copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["risk"] = df["predicted_label"].map(lambda e: EMOTION_META[e]["risk"])

    total = len(df)

    # -- KPIs --------------------------------------------------------------- #
    high_risk = int((df["risk"] >= 3).sum())
    at_risk = int((df["risk"] >= 2).sum())
    stable = int((df["risk"] == 0).sum())
    kpis = {
        "total": total,
        "high_risk": high_risk,
        "high_risk_pct": round(100 * high_risk / total, 1),
        "at_risk_pct": round(100 * at_risk / total, 1),
        "stable_pct": round(100 * stable / total, 1),
        "avg_confidence": round(100 * df["confidence"].mean(), 1),
        "subreddit_count": int(df["subreddit"].nunique()),
    }

    # -- Emotion distribution (donut) -------------------------------------- #
    counts = df["predicted_label"].value_counts()
    emotion_distribution = [{
        "emotion": e,
        "count": int(counts.get(e, 0)),
        "pct": round(100 * counts.get(e, 0) / total, 1),
        "color": EMOTION_META[e]["color"],
        "emoji": EMOTION_META[e]["emoji"],
        "risk": EMOTION_META[e]["risk"],
    } for e in LABELS]
    emotion_distribution.sort(key=lambda d: d["count"], reverse=True)

    # -- Timeline (per-day emotion counts) --------------------------------- #
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    timeline = []
    for day, g in df.groupby("day"):
        c = g["predicted_label"].value_counts().to_dict()
        timeline.append({
            "date": day,
            "total": int(len(g)),
            "counts": {e: int(c.get(e, 0)) for e in LABELS},
            "at_risk": int((g["risk"] >= 2).sum()),
            "high_risk": int((g["risk"] >= 3).sum()),
        })
    timeline.sort(key=lambda d: d["date"])

    # -- Per-subreddit breakdown ------------------------------------------- #
    by_subreddit = []
    for sub, g in df.groupby("subreddit"):
        c = g["predicted_label"].value_counts().to_dict()
        n = len(g)
        by_subreddit.append({
            "subreddit": sub,
            "total": int(n),
            "counts": {e: int(c.get(e, 0)) for e in LABELS},
            "stable_pct": round(100 * (g["risk"] == 0).sum() / n, 1),
            "at_risk_pct": round(100 * (g["risk"] >= 2).sum() / n, 1),
            "high_risk_pct": round(100 * (g["risk"] >= 3).sum() / n, 1),
        })
    by_subreddit.sort(key=lambda d: d["total"], reverse=True)

    # -- Sources ----------------------------------------------------------- #
    sources = {k: int(v) for k, v in df["source"].value_counts().items()}

    # -- Confidence buckets ------------------------------------------------ #
    bins = [0, 0.4, 0.55, 0.7, 0.85, 1.01]
    labels = ["<40%", "40-55%", "55-70%", "70-85%", "85-100%"]
    cut = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    confidence_buckets = [{"range": l, "count": int((cut == l).sum())} for l in labels]

    # -- Topic cloud (term -> count + dominant emotion colour) ------------- #
    term_counts = Counter()
    term_emotion = defaultdict(Counter)
    for text, emo in zip(df["text"].astype(str), df["predicted_label"]):
        toks = set(t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP)
        for t in toks:
            term_counts[t] += 1
            term_emotion[t][emo] += 1
    topics = []
    for term, cnt in term_counts.most_common(60):
        dom = term_emotion[term].most_common(1)[0][0]
        topics.append({"term": term, "count": int(cnt),
                       "emotion": dom, "color": EMOTION_META[dom]["color"]})

    # -- Top mentions (high score) + high-risk samples --------------------- #
    def row_to_mention(r):
        e = r["predicted_label"]
        return {
            "text": snippet(r["text"]),
            "subreddit": r["subreddit"],
            "source": r["source"],
            "emotion": e,
            "emoji": EMOTION_META[e]["emoji"],
            "color": EMOTION_META[e]["color"],
            "risk": int(EMOTION_META[e]["risk"]),
            "confidence": round(100 * float(r["confidence"]), 1),
            "score": int(r["score"]),
            "date": r["date"].strftime("%Y-%m-%d"),
        }

    df["len"] = df["text"].str.len()
    readable = df[(df["len"] > 60) & (df["len"] < 1200)]
    top_mentions = [row_to_mention(r) for _, r in
                    readable.sort_values("score", ascending=False).head(14).iterrows()]
    high_risk_samples = [row_to_mention(r) for _, r in
                         readable[readable["risk"] >= 3]
                         .sort_values("confidence", ascending=False).head(10).iterrows()]

    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "date_start": df["date"].min().strftime("%Y-%m-%d"),
            "date_end": df["date"].max().strftime("%Y-%m-%d"),
            "subreddits": sorted(df["subreddit"].unique().tolist()),
            "model": MODEL_NOTE,
            "topic": "AI layoffs · tech-worker mental-health signals",
        },
        "kpis": kpis,
        "leaderboard": build_leaderboard(),
        "emotion_distribution": emotion_distribution,
        "timeline": timeline,
        "by_subreddit": by_subreddit,
        "sources": sources,
        "confidence_buckets": confidence_buckets,
        "topics": topics,
        "top_mentions": top_mentions,
        "high_risk_samples": high_risk_samples,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ok] {os.path.relpath(OUT, ROOT)}  | {total:,} mentions "
          f"| {len(timeline)} days | {len(topics)} topics "
          f"| {len(top_mentions)} top mentions")


if __name__ == "__main__":
    main()

"""
SQLite store for anonymous community submissions from the Emotion Partner.

Each interaction is aggregated back into the main dashboard. No accounts, no
identifiers — only the analysis result plus a short snippet used for the live
feed. High-risk (Critical) message text is never exposed in the public feed,
though it still counts toward the aggregates.

On Railway the container filesystem is ephemeral; mount a volume and point
DB_PATH at it to persist community data across deploys.
"""
from __future__ import annotations

import os
import json
import sqlite3
import threading
from datetime import datetime, timezone

from inference import LABELS, EMOTION_META, RISK_TIERS

_WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(_WEBAPP_DIR, "data", "community.db"))
_WRITE_LOCK = threading.Lock()
SNIPPET_MAX = 160


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL = concurrent readers during writes; busy_timeout avoids "locked" errors.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                day         TEXT    NOT NULL,
                snippet     TEXT,
                emotion     TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                risk        INTEGER NOT NULL,
                model       TEXT,
                probs       TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_day ON submissions(day)")
        conn.commit()


def insert_submission(text: str, analysis: dict) -> None:
    """Persist one analysis result for dashboard aggregation."""
    risk = int(analysis.get("risk_level", 0))
    # Privacy/safety: never store or expose raw text for critical-risk messages.
    if risk >= 4:
        snippet = ""
    else:
        snippet = " ".join((text or "").split())[:SNIPPET_MAX]
    now = datetime.now(timezone.utc)
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO submissions (ts, day, snippet, emotion, confidence, "
            "risk, model, probs) VALUES (?,?,?,?,?,?,?,?)",
            (now.isoformat(), now.strftime("%Y-%m-%d"), snippet,
             analysis["top_emotion"], float(analysis["confidence"]), risk,
             analysis.get("model", ""), json.dumps(analysis.get("probabilities", {}))),
        )
        conn.commit()


def get_community_aggregate(recent_limit: int = 24) -> dict:
    """Aggregate community submissions in the same shape as the Reddit data."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM submissions").fetchall()

    total = len(rows)
    counts = {e: 0 for e in LABELS}
    conf_sum = 0.0
    high_risk = at_risk = stable = 0
    day_counts: dict = {}
    model_counts: dict = {}

    for r in rows:
        emo = r["emotion"]
        if emo in counts:
            counts[emo] += 1
        conf_sum += r["confidence"]
        risk = r["risk"]
        high_risk += risk >= 3
        at_risk += risk >= 2
        stable += risk == 0
        day_counts.setdefault(r["day"], {e: 0 for e in LABELS})
        if emo in counts:
            day_counts[r["day"]][emo] += 1
        model_counts[r["model"]] = model_counts.get(r["model"], 0) + 1

    emotion_distribution = [{
        "emotion": e, "count": counts[e],
        "pct": round(100 * counts[e] / total, 1) if total else 0.0,
        "color": EMOTION_META[e]["color"], "emoji": EMOTION_META[e]["emoji"],
        "risk": EMOTION_META[e]["risk"],
    } for e in LABELS]
    emotion_distribution.sort(key=lambda d: d["count"], reverse=True)

    timeline = [{
        "date": day, "total": sum(c.values()),
        "counts": c,
        "at_risk": sum(v for e, v in c.items() if EMOTION_META[e]["risk"] >= 2),
        "high_risk": sum(v for e, v in c.items() if EMOTION_META[e]["risk"] >= 3),
    } for day, c in sorted(day_counts.items())]

    recent = []
    for r in reversed(rows[-recent_limit:]):
        emo = r["emotion"]
        meta = EMOTION_META[emo]
        recent.append({
            "text": r["snippet"] if r["risk"] < 4 else "[hidden] high-risk message — withheld for safety",
            "emotion": emo, "emoji": meta["emoji"], "color": meta["color"],
            "risk": r["risk"], "risk_tier": RISK_TIERS.get(r["risk"], ""),
            "confidence": round(r["confidence"], 1), "model": r["model"],
            "ts": r["ts"],
        })

    return {
        "kpis": {
            "total": total,
            "high_risk": high_risk,
            "high_risk_pct": round(100 * high_risk / total, 1) if total else 0.0,
            "at_risk_pct": round(100 * at_risk / total, 1) if total else 0.0,
            "stable_pct": round(100 * stable / total, 1) if total else 0.0,
            "avg_confidence": round(conf_sum / total, 1) if total else 0.0,
        },
        "emotion_distribution": emotion_distribution,
        "timeline": timeline,
        "recent": recent,
        "by_model": model_counts,
    }

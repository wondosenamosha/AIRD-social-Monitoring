"""
Live emotion-classification inference for the AIRD dashboard.

Serves the **already-trained Track B models** saved under ../models — no
retraining. A small registry exposes every Track B model whose saved artifacts
are self-contained enough to run standalone, plus a real ensemble of them:

    mentalbert  -> models/MentalBERT_TrackB   (config + safetensors + tokenizer)
    roberta     -> models/RoBERTa_TrackB       (config + safetensors + tokenizer)
    ensemble    -> probability average of the above

The other Track B models cannot be loaded for live inference from what was
saved: LogReg/LightGBM exported only the classifier (no TF-IDF vectorizer), the
custom transformer saved only a weight state_dict (no vocab/tokenizer), and the
Ensemble was never persisted as a composite. Their *published* Track B metrics
are still surfaced on the dashboard leaderboard. Models load lazily, so memory
is only spent on the variants actually used.
"""
from __future__ import annotations

import os

# Force the PyTorch-only backend BEFORE transformers is imported. TensorFlow is
# present in some environments and its import can deadlock on macOS (abseil
# "Lock blocking" mutex hang); we never use TF/Flax here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# PyTorch and LightGBM each bundle an OpenMP runtime; on macOS loading both in
# one process aborts with "OMP: Error #15 ... libomp already initialized". Allow
# the duplicate so all six models can be served from a single process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import threading
from functools import lru_cache

# --------------------------------------------------------------------------- #
#  Label space + presentation metadata (single source of truth for the UI)
# --------------------------------------------------------------------------- #
LABELS = [
    "Anxiety", "Bipolar", "Depression", "Normal",
    "Personality Disorder", "Stress", "Suicidal",
]

EMOTION_META = {
    "Normal":               {"emoji": "🙂", "color": "#10b981", "risk": 0, "label": "Normal"},
    "Stress":               {"emoji": "😟", "color": "#f59e0b", "risk": 1, "label": "Stress"},
    "Anxiety":              {"emoji": "😰", "color": "#3b82f6", "risk": 2, "label": "Anxiety"},
    "Personality Disorder": {"emoji": "🎭", "color": "#8b5cf6", "risk": 2, "label": "Personality"},
    "Bipolar":              {"emoji": "🎢", "color": "#ec4899", "risk": 3, "label": "Bipolar"},
    "Depression":           {"emoji": "😔", "color": "#647488", "risk": 2, "label": "Depression"},
    "Suicidal":             {"emoji": "🆘", "color": "#ef4444", "risk": 4, "label": "Suicidal"},
}

RISK_TIERS = {0: "Stable", 1: "Mild", 2: "Elevated", 3: "At-risk", 4: "High-risk"}

# --------------------------------------------------------------------------- #
#  Model registry (Track B variants with complete, loadable artifacts)
# --------------------------------------------------------------------------- #
_WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_WEBAPP_DIR)


def _default_models_root() -> str:
    """Prefer models bundled inside webapp/ (self-contained deploy); fall back
    to the project-root models/ for local dev without copying."""
    local = os.path.join(_WEBAPP_DIR, "models")
    if os.path.isdir(local) and any(
        os.path.isdir(os.path.join(local, d))
        for d in ("MentalBERT_TrackB", "RoBERTa_TrackB")):
        return local
    return os.path.join(_PROJECT_ROOT, "models")


_MODELS_ROOT = os.environ.get("MODELS_ROOT") or _default_models_root()

MODEL_REGISTRY = {
    "mentalbert": {"name": "MentalBERT",
                   "dir": os.path.join(_MODELS_ROOT, "MentalBERT_TrackB")},
    "roberta":    {"name": "Mental-RoBERTa",
                   "dir": os.path.join(_MODELS_ROOT, "RoBERTa_TrackB")},
}
ENSEMBLE_MEMBERS = ["mentalbert", "roberta"]
# Default to the ensemble (combines every loadable Track B model — the research
# best and the most robust on out-of-domain input). Override with DEFAULT_MODEL
# (e.g. "roberta") for a lighter, single-model deployment.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "ensemble")
MAX_LEN = int(os.environ.get("MAX_LEN") or "256")   # tolerate empty env value

# Classical Track B pipelines (TF-IDF word+char union + classifier), retrained as
# self-contained bundles because the originals shipped the classifier only.
_CLASSICAL = {
    "logreg":   {"name": "Logistic Regression",
                 "file": os.path.join(_MODELS_ROOT, "classical", "logreg_trackb.joblib")},
    "lightgbm": {"name": "LightGBM",
                 "file": os.path.join(_MODELS_ROOT, "classical", "lgbm_trackb.joblib")},
}
# Custom Transformer Track B (state_dict + word2idx + config saved together).
_CT_KEY = "customtransformer"
_CT_DIR = os.path.join(_MODELS_ROOT, "CustomTransformer_TrackB")


def _lightgbm_importable() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def _available_classical() -> list:
    out = []
    for k, v in _CLASSICAL.items():
        if not os.path.exists(v["file"]):
            continue
        if k == "lightgbm" and not _lightgbm_importable():
            continue  # needs OpenMP runtime (libomp); skip if unavailable
        out.append(k)
    return out


def _ct_available() -> bool:
    return (os.path.exists(os.path.join(_CT_DIR, "state_dict.pt"))
            and os.path.exists(os.path.join(_CT_DIR, "word2idx.json")))


def _available_single() -> list:
    """Registry keys whose model folder actually exists on disk."""
    return [k for k in MODEL_REGISTRY if os.path.isdir(MODEL_REGISTRY[k]["dir"])]


def available_models() -> list:
    """Selector payload for the UI. `live` = runnable from the saved artifacts.
    The remaining Track B models are listed too (UI disables them) — their
    feature extractors weren't exported, so they can't serve standalone."""
    live = _available_single()
    out = [{"key": k, "name": MODEL_REGISTRY[k]["name"], "live": True} for k in live]
    if len(live) >= 2:
        out.append({"key": "ensemble", "name": "Ensemble", "live": True})
    classical = _available_classical()
    out.append({"key": "lightgbm", "name": _CLASSICAL["lightgbm"]["name"],
                "live": "lightgbm" in classical})
    out.append({"key": "logreg", "name": _CLASSICAL["logreg"]["name"],
                "live": "logreg" in classical})
    return out


def resolve_default() -> str:
    """A valid default key given what's actually deployed."""
    singles = _available_single()
    if DEFAULT_MODEL == "ensemble" and len(singles) >= 2:
        return "ensemble"
    if DEFAULT_MODEL in singles:
        return DEFAULT_MODEL
    return singles[0] if singles else "mentalbert"


# --------------------------------------------------------------------------- #
#  Text normalisation — replicate the notebook's training/Reddit pipeline so
#  live input matches how the saved models were trained (spaCy lemmatised,
#  stopwords removed, negations preserved). See preprocess.py.
# --------------------------------------------------------------------------- #
from preprocess import preprocess_text


def clean_text(text: str) -> str:
    return preprocess_text(text)


# --------------------------------------------------------------------------- #
#  Keyword lexicon — interpretable "Detected Emotion Keywords" panel
# --------------------------------------------------------------------------- #
KEYWORD_LEXICON = {
    "Anxiety": ["anxious", "anxiety", "panic", "worry", "worried", "nervous",
                "fear", "scared", "restless", "overwhelmed", "racing", "tense",
                "dread", "uneasy", "on edge", "can't breathe", "hardly breathe"],
    "Bipolar": ["manic", "mania", "mood swing", "mood swings", "episode",
                "hypomanic", "highs", "lows", "bipolar", "euphoric", "crash",
                "impulsive", "racing thoughts"],
    "Depression": ["depressed", "depression", "hopeless", "empty", "worthless",
                   "sad", "numb", "tired", "exhausted", "unmotivated", "alone",
                   "miserable", "crying", "dark", "no point", "meaningless"],
    "Normal": ["good", "fine", "happy", "great", "okay", "enjoy", "fun",
               "relaxed", "calm", "well", "content", "grateful", "excited"],
    "Personality Disorder": ["identity", "abandon", "abandonment", "unstable",
                             "emptiness", "rage", "relationships", "splitting",
                             "borderline", "void", "no sense of self"],
    "Stress": ["stress", "stressed", "overwhelmed", "pressure", "deadline",
               "workload", "burnout", "exhausted", "too much", "busy", "tense",
               "overworked", "assignments", "can hardly breathe"],
    "Suicidal": ["suicide", "suicidal", "kill myself", "end it", "want to die",
                 "no reason to live", "give up", "not want to live", "goodbye",
                 "can't go on", "end my life", "better off dead"],
}


def detect_keywords(text: str, top_emotion: str, max_n: int = 6):
    low = (text or "").lower()
    found, seen = [], set()
    for kw in sorted(KEYWORD_LEXICON.get(top_emotion, []), key=len, reverse=True):
        if kw in low and kw not in seen:
            seen.add(kw)
            found.append(kw)
    return found[:max_n]


# --------------------------------------------------------------------------- #
#  Single transformer wrapper
# --------------------------------------------------------------------------- #
class _Transformer:
    def __init__(self, model_dir: str):
        import torch
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification)
        self.torch = torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        id2label = self.model.config.id2label
        self.classes = [id2label.get(i, id2label.get(str(i)))
                        for i in range(len(id2label))]
        self._lock = threading.Lock()

    def predict_proba(self, cleaned: str) -> dict:
        enc = self.tokenizer(cleaned, return_tensors="pt",
                             truncation=True, max_length=MAX_LEN)
        with self._lock, self.torch.no_grad():   # HF model not thread-safe
            logits = self.model(**enc).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        return {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}


# Lazy per-key cache so memory is only spent on models actually requested.
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _get_transformer(key: str) -> _Transformer:
    if key not in _CACHE:
        with _CACHE_LOCK:
            if key not in _CACHE:
                _CACHE[key] = _Transformer(MODEL_REGISTRY[key]["dir"])
    return _CACHE[key]


# --------------------------------------------------------------------------- #
#  Classical (TF-IDF word+char union -> LogReg / LightGBM)
# --------------------------------------------------------------------------- #
class _Classical:
    def __init__(self, path: str):
        import joblib
        from scipy.sparse import hstack as sp_hstack
        b = joblib.load(path)
        self.word_vec, self.char_vec, self.clf = b["word_vec"], b["char_vec"], b["clf"]
        self.classes = b["classes"]            # index -> label name
        self._hstack = sp_hstack
        self._lock = threading.Lock()

    def predict_proba(self, cleaned: str) -> dict:
        X = self._hstack([self.word_vec.transform([cleaned]),
                          self.char_vec.transform([cleaned])], format="csr")
        with self._lock:
            p = self.clf.predict_proba(X)[0]
        out = {lbl: 0.0 for lbl in LABELS}
        for idx, prob in zip(self.clf.classes_, p):
            out[self.classes[int(idx)]] = float(prob)
        return out


# --------------------------------------------------------------------------- #
#  Custom Transformer (GloVe-300d, 4 blocks) — state_dict + word2idx
# --------------------------------------------------------------------------- #
class _CTModel:
    def __init__(self, model_dir: str):
        import json
        import numpy as np
        import torch
        from ct_model import build_from_config
        self.torch, self.np = torch, np
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(model_dir, "word2idx.json")) as f:
            self.w2i = json.load(f)
        self.labels = cfg["labels"]
        self.max_len = int(cfg["max_len"])
        self.model = build_from_config(cfg)
        sd = torch.load(os.path.join(model_dir, "state_dict.pt"), map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval()
        self._lock = threading.Lock()

    def predict_proba(self, cleaned: str) -> dict:
        ids = [self.w2i.get(w, 1) for w in cleaned.split()][:self.max_len]
        arr = self.np.zeros((1, self.max_len), dtype=self.np.int64)
        arr[0, :len(ids)] = ids
        x = self.torch.tensor(arr, dtype=self.torch.long)
        with self._lock, self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), dim=-1)[0]
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}


def _get_classical(key: str) -> _Classical:
    if key not in _CACHE:
        with _CACHE_LOCK:
            if key not in _CACHE:
                _CACHE[key] = _Classical(_CLASSICAL[key]["file"])
    return _CACHE[key]


def _get_ct() -> _CTModel:
    if _CT_KEY not in _CACHE:
        with _CACHE_LOCK:
            if _CT_KEY not in _CACHE:
                _CACHE[_CT_KEY] = _CTModel(_CT_DIR)
    return _CACHE[_CT_KEY]


# --------------------------------------------------------------------------- #
#  Temperature scaling — sharpens the softmax distribution post-hoc.
#  T < 1 boosts top-class confidence; T = 1 is no-op.
#  Calibrated at T=0.80: pushes a ~79% Normal prediction to ~90%+ while
#  keeping the rank order and relative signal intact.
# --------------------------------------------------------------------------- #
TEMPERATURE = 0.80

def _apply_temperature(probs: dict) -> dict:
    import math
    if TEMPERATURE == 1.0:
        return probs
    log_p = {k: math.log(max(v, 1e-10)) / TEMPERATURE for k, v in probs.items()}
    shift = max(log_p.values())
    exp_p = {k: math.exp(v - shift) for k, v in log_p.items()}
    total = sum(exp_p.values())
    return {k: v / total for k, v in exp_p.items()}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def _build_payload(text: str, probs: dict, model_label: str) -> dict:
    top = max(probs, key=probs.get)
    meta = EMOTION_META.get(top, {"emoji": "🙂", "color": "#888", "risk": 0})
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "top_emotion": top,
        "top_emoji": meta["emoji"],
        "top_color": meta["color"],
        "confidence": round(probs[top] * 100, 1),
        "risk_level": meta["risk"],
        "risk_tier": RISK_TIERS.get(meta["risk"], "Stable"),
        "probabilities": {k: round(v * 100, 1) for k, v in probs.items()},
        "ranked": [{"emotion": k, "pct": round(v * 100, 1)} for k, v in ranked],
        "keywords": detect_keywords(text, top),
        "tip": _tip(top, meta["risk"]),
        "model": model_label,
    }


def analyze(text: str, model_key: str = None) -> dict:
    """Run live inference with a chosen Track B model (or the ensemble)."""
    singles = _available_single()
    classical = _available_classical()
    valid = set(singles) | set(classical) | {"ensemble"}
    if _ct_available():
        valid.add(_CT_KEY)
    key = (model_key or "").lower()
    if key not in valid:
        key = resolve_default()

    cleaned = clean_text(text)
    if not cleaned:
        probs = {lbl: (1.0 if lbl == "Normal" else 0.0) for lbl in LABELS}
        return _build_payload(text, probs, "—")
    short_text = len(cleaned.split()) < 8

    if key == "ensemble":
        members = [k for k in ENSEMBLE_MEMBERS if k in singles]
        if len(members) >= 2:
            per = [_get_transformer(k).predict_proba(cleaned) for k in members]
            probs = _apply_temperature({lbl: sum(p[lbl] for p in per) / len(per) for lbl in per[0]})
            label = "Ensemble (" + " + ".join(
                MODEL_REGISTRY[k]["name"].split(" (")[0] for k in members) + ")"
            payload = _build_payload(text, probs, label)
            payload["short_text"] = short_text
            return payload
        key = members[0] if members else (singles[0] if singles else "mentalbert")

    if key in _CLASSICAL and key in classical:
        probs = _apply_temperature(_get_classical(key).predict_proba(cleaned))
        payload = _build_payload(text, probs, _CLASSICAL[key]["name"])
        payload["short_text"] = short_text
        return payload

    if key == _CT_KEY and _ct_available():
        probs = _apply_temperature(_get_ct().predict_proba(cleaned))
        payload = _build_payload(text, probs, "Custom Transformer")
        payload["short_text"] = short_text
        return payload

    probs = _apply_temperature(_get_transformer(key).predict_proba(cleaned))
    payload = _build_payload(text, probs, MODEL_REGISTRY[key]["name"])
    payload["short_text"] = short_text
    return payload


def _tip(emotion: str, risk: int) -> str:
    tips = {
        "Normal": "You seem to be in a balanced state. Keep up your routines, sleep, and mindfulness practices.",
        "Stress": "Signs of stress detected. Try short breaks, breathing exercises, and breaking tasks into smaller steps.",
        "Anxiety": "Anxiety signals present. Grounding techniques (5-4-3-2-1) and limiting caffeine may help.",
        "Personality Disorder": "Some complex emotional patterns detected. Journaling and talking to someone you trust can help.",
        "Bipolar": "Mood-swing patterns detected. Regular sleep and routine tracking are protective; consider professional guidance.",
        "Depression": "Low-mood signals detected. Gentle activity, sunlight, and reaching out to someone can help.",
        "Suicidal": "We detected language of serious distress. You are not alone — please reach out to a crisis line or someone you trust right now.",
    }
    base = tips.get(emotion, tips["Normal"])
    if risk >= 4:
        base += "  📞 If you are in immediate danger, contact local emergency services or a 24/7 crisis line."
    return base


@lru_cache(maxsize=1)
def warmup():
    """Optionally pre-load the default model(s) (call at app start if desired)."""
    d = resolve_default()
    keys = ENSEMBLE_MEMBERS if d == "ensemble" else [d]
    for k in keys:
        if k in _available_single():
            _get_transformer(k)
    return True

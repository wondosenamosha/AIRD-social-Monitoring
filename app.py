"""
AIRD Social Monitoring — Flask backend (production-hardened).

Serves an Awario-style public dashboard built from the existing Reddit
inference results, plus an interactive "Emotion Partner" that runs the saved
Track B models live. Every interaction is aggregated back into the dashboard.

Production features: security headers + CSP, real client-IP handling behind a
proxy, per-IP rate limiting on the heavy endpoints, request-size limits, JSON
error handlers, structured logging, and a health check.

Routes
------
GET  /                 dashboard + Emotion Partner UI
GET  /api/dashboard    Reddit baseline aggregates + live community aggregates
GET  /api/community    live community aggregates only (for refresh/polling)
GET  /api/models       Track B models available for live inference
POST /api/analyze      run live inference, store submission, return analysis
POST /api/analyze_url  scrape a Reddit post, analyse author + comments
GET  /healthz          health check (Railway)
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
from collections import deque

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

import db
import inference
import reddit_scrape

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
_WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
REDDIT_JSON = os.path.join(_WEBAPP_DIR, "data", "reddit_dashboard.json")
MAX_CHARS = int(os.environ.get("MAX_CHARS") or "5000")    # tolerate empty env value
RATE_LIMIT = int(os.environ.get("RATE_LIMIT") or "30")     # requests / window
RATE_WINDOW = int(os.environ.get("RATE_WINDOW") or "60")   # seconds

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("aird")

CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'self'; frame-ancestors 'none'; object-src 'none'"
)
# Minimal inline SVG favicon (avoids a 404 and needs no static file).
FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='#2bb3c0'/>"
    "<text x='16' y='22' font-family='Inter,Arial' font-size='16' "
    "font-weight='700' fill='#fff' text-anchor='middle'>A</text></svg>"
)

app = Flask(__name__, template_folder="templates", static_folder="static")
# Honour X-Forwarded-* from Railway's proxy so rate limiting sees real IPs.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024            # JSON payloads are tiny
app.config["JSON_SORT_KEYS"] = False

# Load the (static) Reddit baseline once at startup.
with open(REDDIT_JSON) as f:
    REDDIT_DATA = json.load(f)

db.init_db()

if os.environ.get("WARMUP", "0") == "1":
    try:
        inference.warmup()
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("warmup failed: %s", exc)


# --------------------------------------------------------------------------- #
#  Lightweight in-process per-IP rate limiter (single worker + threads)
# --------------------------------------------------------------------------- #
_RL: dict = {}
_RL_LOCK = threading.Lock()


def _rate_ok(ip: str):
    """Sliding-window limiter. Returns (allowed, retry_after_seconds)."""
    now = time.time()
    cutoff = now - RATE_WINDOW
    with _RL_LOCK:
        q = _RL.get(ip)
        if q is None:
            q = _RL[ip] = deque()
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return False, int(q[0] + RATE_WINDOW - now) + 1
        q.append(now)
        if len(_RL) > 5000:                              # crude memory guard
            for k in [k for k, v in _RL.items() if not v]:
                _RL.pop(k, None)
        return True, 0


def _rate_limited():
    ok, retry = _rate_ok(request.remote_addr or "anon")
    if ok:
        return None
    resp = jsonify({"error": "Too many requests — please slow down and retry shortly."})
    resp.status_code = 429
    resp.headers["Retry-After"] = str(retry)
    return resp


# --------------------------------------------------------------------------- #
#  Security headers
# --------------------------------------------------------------------------- #
@app.after_request
def _security_headers(resp: Response) -> Response:
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    resp.headers.setdefault("Content-Security-Policy", CSP)
    return resp


# --------------------------------------------------------------------------- #
#  Pages / static
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template(
        "index.html",
        meta=REDDIT_DATA["meta"],
        models=inference.available_models(),
        default_model=inference.resolve_default(),
    )


@app.route("/favicon.ico")
def favicon():
    return Response(FAVICON, mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------------------------------------------------- #
#  Read APIs
# --------------------------------------------------------------------------- #
@app.route("/api/dashboard")
def api_dashboard():
    return jsonify({"reddit": REDDIT_DATA, "community": db.get_community_aggregate()})


@app.route("/api/community")
def api_community():
    return jsonify(db.get_community_aggregate())


@app.route("/api/models")
def api_models():
    return jsonify({"models": inference.available_models(),
                    "default": inference.resolve_default()})


# --------------------------------------------------------------------------- #
#  Write APIs (rate limited)
# --------------------------------------------------------------------------- #
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    limited = _rate_limited()
    if limited:
        return limited
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    model_key = payload.get("model")
    if not text:
        return jsonify({"error": "Please describe how you're feeling."}), 400
    text = text[:MAX_CHARS]

    try:
        analysis = inference.analyze(text, model_key)
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found under models/. "
                                 "Check MODELS_ROOT."}), 503
    except Exception as exc:
        log.exception("inference failed")
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    try:
        db.insert_submission(text, analysis)
    except Exception:
        log.exception("failed to persist submission")

    return jsonify({"analysis": analysis, "community": db.get_community_aggregate()})


@app.route("/api/analyze_url", methods=["POST"])
def api_analyze_url():
    """Scrape a Reddit post, analyse the author + a slice of comments, and
    aggregate everything into the dashboard."""
    limited = _rate_limited()
    if limited:
        return limited
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    model_key = payload.get("model")
    if not url:
        return jsonify({"error": "Please paste a Reddit post link."}), 400

    try:
        post = reddit_scrape.fetch_post(url)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    op_text = (post["title"] + "\n\n" + post["selftext"]).strip()
    if len(op_text) < 3:
        return jsonify({"error": "This post has no analyzable text "
                                 "(image- or link-only post)."}), 422
    op_text = op_text[:MAX_CHARS]

    try:
        analysis = inference.analyze(op_text, model_key)
    except Exception as exc:
        log.exception("inference failed")
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    try:
        db.insert_submission(op_text, analysis)
    except Exception:
        log.exception("failed to persist submission")

    counts, comment_rows = {}, []
    for c in post["comments"]:
        try:
            ca = inference.analyze(c["body"], model_key)
        except Exception:
            continue
        counts[ca["top_emotion"]] = counts.get(ca["top_emotion"], 0) + 1
        try:
            db.insert_submission(c["body"], ca)
        except Exception:
            pass
        comment_rows.append({
            "text": ("[hidden] high-risk comment — withheld for safety"
                     if ca["risk_level"] >= 4 else _snip(c["body"], 160)),
            "emotion": ca["top_emotion"], "emoji": ca["top_emoji"],
            "color": ca["top_color"], "confidence": ca["confidence"],
            "risk": ca["risk_level"], "score": c["score"],
        })

    distribution = [{"emotion": e, "count": n,
                     "color": inference.EMOTION_META[e]["color"],
                     "emoji": inference.EMOTION_META[e]["emoji"]}
                    for e, n in sorted(counts.items(), key=lambda kv: -kv[1])]

    return jsonify({
        "analysis": analysis,
        "post": {
            "title": post["title"], "subreddit": post["subreddit"],
            "author": post["author"], "score": post["score"],
            "num_comments": post["num_comments"], "permalink": post["permalink"],
            "snippet": _snip(post["selftext"] or post["title"], 280),
        },
        "thread": {"analyzed": len(comment_rows), "distribution": distribution,
                   "comments": comment_rows[:6]},
        "community": db.get_community_aggregate(),
    })


# --------------------------------------------------------------------------- #
#  Health + errors
# --------------------------------------------------------------------------- #
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok",
                    "models": [m["key"] for m in inference.available_models() if m["live"]]})


@app.errorhandler(404)
def _e404(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def _e413(_):
    return jsonify({"error": "Payload too large"}), 413


@app.errorhandler(500)
def _e500(exc):
    log.exception("unhandled 500: %s", exc)
    return jsonify({"error": "Internal server error"}), 500


def _snip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("…" if len(text) > n else "")


if __name__ == "__main__":
    port = int(os.environ.get("PORT") or "8000")
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("DEBUG")))

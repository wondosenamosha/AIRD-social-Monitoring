# AIRD · Social Monitoring Dashboard

A public, Awario-style social-monitoring dashboard for **mental-health signals**,
built from the project's Reddit inference results, plus an interactive
**Emotion Partner** that runs the saved **Track B** models live. Every
interaction is aggregated back into the dashboard (Reddit → Community → Combined).

This `webapp/` folder is **self-contained and deployable** (Railway-ready).

---

## What's inside

```
webapp/
├── app.py                     # Flask backend (APIs + page)
├── inference.py               # Live inference over saved Track B models (+ ensemble)
├── preprocess.py              # Exact notebook preprocessing (spaCy + nltk + contractions)
├── db.py                      # SQLite store for anonymous community submissions
├── models/                    # Loadable Track B models (copied from ../models)
│   ├── MentalBERT_TrackB/
│   └── RoBERTa_TrackB/
├── data/
│   └── reddit_dashboard.json  # Precomputed Reddit aggregates + Track B leaderboard
├── scripts/
│   └── build_reddit_aggregates.py   # Rebuild data/reddit_dashboard.json
├── templates/index.html       # Dashboard + Emotion Partner UI
├── static/css/style.css
├── static/js/{dashboard.js,partner.js}
├── requirements.txt           # Lean serving deps (PyTorch installed via nixpacks)
├── nixpacks.toml              # Railway build (CPU torch + spaCy model + nltk data)
├── railway.json               # Railway deploy config (healthcheck etc.)
└── Procfile                   # gunicorn entrypoint
```

### Analyze a Reddit link 🔗
In the Emotion Partner you can **paste a Reddit post URL** (or just type how you
feel). The app scrapes the post + comments, analyzes the author's emotional
state (4-screen report) and the thread, and aggregates everything into the
dashboard. Reddit blocks anonymous `.json`/proxy access, so scraping uses
Reddit's public **RSS feed** (`…/comments/<id>/.rss`) — credential-free and
works for recent posts. If `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` are set it
also uses PRAW (adds post score). See `reddit_scrape.py`.

### Live models (Track B)
| Model | Live | Why |
|---|---|---|
| **MentalBERT (Track B)** | ✅ | full HF artifacts (config + safetensors + tokenizer) |
| **Mental-RoBERTa (Track B)** | ✅ | full HF artifacts |
| **Ensemble** | ✅ | probability average of the two above |
| LightGBM / LogReg (Track B) | ❌ | classifier saved without its TF-IDF vectorizer |
| Custom Transformer (Track B) | ❌ | only a weight `state_dict` (no vocab/tokenizer) |

All six Track B models' **published** metrics still appear on the dashboard leaderboard.

---

## Run locally

```bash
cd webapp
pip install -r requirements.txt
pip install torch                      # CPU build is fine
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

python app.py                          # http://localhost:8000
```

The first `/api/analyze` call loads the transformer (~15–40s); subsequent calls are fast.

### Rebuild the Reddit aggregates (optional)
```bash
python scripts/build_reddit_aggregates.py   # reads ../results/reddit_inference_results.csv
```

---

## Deploy to Railway

The model files exceed GitHub's 100 MB limit, so deploy with the **Railway CLI**
(`railway up`), which uploads local files directly.

```bash
npm i -g @railway/cli
railway login
cd webapp           # deploy THIS folder as the project root
railway init
railway up
```

Railway auto-detects `nixpacks.toml`, which installs CPU-only PyTorch, the
serving deps, the spaCy model, and nltk data. `railway.json` sets the start
command and `/healthz` health check.

**Memory:** MentalBERT alone needs ~1.5 GB RAM; loading the Ensemble (both
transformers) needs ~3 GB. Pick a Railway plan accordingly. Set `DEFAULT_MODEL`
(`mentalbert` default) and optionally `WARMUP=1` to preload at boot.

**Persistence:** community submissions live in SQLite (`data/community.db`).
Railway's filesystem is ephemeral — mount a volume and set `DB_PATH` to keep
them across deploys.

### Environment variables
| Var | Default | Purpose |
|---|---|---|
| `DEFAULT_MODEL` | `ensemble` | model used when none selected (`mentalbert`/`roberta` are lighter) |
| `MODELS_ROOT` | `webapp/models` | where the Track B model folders live |
| `DB_PATH` | `webapp/data/community.db` | SQLite location (use a volume on Railway) |
| `WARMUP` | `0` | `1` = load default model at boot |
| `RATE_LIMIT` / `RATE_WINDOW` | `30` / `60` | per-IP request cap per window (seconds) on the write APIs |
| `WEB_CONCURRENCY` / `THREADS` / `TIMEOUT` | `1` / `4` / `180` | gunicorn workers / threads / request timeout |
| `MAX_CHARS` | `5000` | max characters analysed per request |
| `LOG_LEVEL` | `INFO` | app + gunicorn log level |
| `PORT` | `8000` | served port (Railway sets this) |

---

## Production hardening

- **Security headers** on every response: CSP (allows only self + the Chart.js
  CDN + Google Fonts), `X-Content-Type-Options`, `X-Frame-Options: DENY`,
  `Referrer-Policy`, `Permissions-Policy`.
- **Rate limiting** (per real client IP via `ProxyFix`) on `/api/analyze` and
  `/api/analyze_url` — protects the expensive model + scrape paths.
- **Request-size limit** (64 KB) and JSON **error handlers** (404/413/429/500).
- **SQLite in WAL mode** with a busy timeout for safe concurrent reads/writes.
- **Health check** at `/healthz`; **gunicorn** config in `gunicorn.conf.py`
  (1 worker × 4 threads, logs to stdout). The model loads lazily so the health
  check passes immediately; set `WARMUP=1` to preload.
- **Live dashboard**: the Community pulse auto-refreshes every 20 s.

## Ethics & safety

This is a **screening / awareness tool, not a diagnosis**. High-risk messages
are never shown in the public Community feed; a crisis-resources panel (988,
Samaritans, findahelpline.com) is always available in the Emotion Partner and a
helpline link sits in the footer. A crisis tip is surfaced for critical-risk
results.

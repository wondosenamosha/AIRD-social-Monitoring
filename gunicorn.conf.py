"""Production gunicorn config for the AIRD dashboard.

One worker by default (the transformer weights are large — avoid loading the
model N times); concurrency comes from threads. A long timeout covers the
first lazy model load. Logs go to stdout/stderr for the platform to capture.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT') or '8000'}"
workers = int(os.environ.get("WEB_CONCURRENCY") or "1")
threads = int(os.environ.get("THREADS") or "4")
worker_class = "gthread"
timeout = int(os.environ.get("TIMEOUT") or "180")      # first model load can be slow
graceful_timeout = 30
keepalive = 5
max_requests = int(os.environ.get("MAX_REQUESTS") or "0")  # 0 = unlimited
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
forwarded_allow_ips = "*"   # trust the platform proxy (Railway) for X-Forwarded-*

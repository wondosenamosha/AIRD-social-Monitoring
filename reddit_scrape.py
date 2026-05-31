"""
Reddit post scraper for the "analyze a link" feature.

Reddit blocks anonymous access to its `.json` API and proxies (HTTP 403), and
credential-free archives (PullPush) don't cover recent posts. But Reddit's
**Atom/RSS feed** for a post (`…/comments/<id>/.rss`) is still served publicly
and includes the post body + comments — so it works for recent content with
**no credentials**. That is the primary path here.

Resolution order:
  1. RSS feed            (credential-free, works for recent posts)   ← primary
  2. PRAW                (only if REDDIT_CLIENT_ID/SECRET are set; adds score)
  3. public `.json`      (rarely works; usually 403)
"""
from __future__ import annotations

import os
import re
import html
import json
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

# RSS feeds are meant for readers; a browser-style UA avoids Reddit's bot block.
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
API_UA = os.environ.get("REDDIT_USER_AGENT",
                        "AIRD-SocialMonitoring/1.0 (mental-health dashboard)")
_NS = {"a": "http://www.w3.org/2005/Atom"}
_POST_RE = re.compile(r"reddit\.com/.+?/comments/[a-z0-9]+", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SUBMITTED_RE = re.compile(r"\s*submitted by\s*/u/\S+.*$", re.IGNORECASE)


def _have_creds() -> bool:
    return bool(os.environ.get("REDDIT_CLIENT_ID")
                and os.environ.get("REDDIT_CLIENT_SECRET"))


def normalize(url: str) -> str:
    url = (url or "").strip().split("#")[0].split("?")[0].rstrip("/")
    if not url:
        raise ValueError("Please paste a Reddit post link.")
    if not _POST_RE.search(url):
        raise ValueError("That doesn't look like a Reddit post link "
                         "(expected …reddit.com/r/<sub>/comments/<id>/…).")
    if not url.startswith("http"):
        url = "https://" + url.lstrip("/")
    return url


def _subreddit(url: str) -> str:
    m = re.search(r"/r/([^/]+)/", url)
    return m.group(1) if m else ""


def _clean(htmltext: str) -> str:
    t = _TAG_RE.sub(" ", htmltext or "")
    t = html.unescape(t)
    t = _SUBMITTED_RE.sub("", t)
    return _WS_RE.sub(" ", t).strip()


def _pack(title, selftext, subreddit, author, score, num_comments, permalink, comments):
    return {"title": title or "", "selftext": selftext or "",
            "subreddit": subreddit or "", "author": author or "",
            "score": score, "num_comments": int(num_comments or 0),
            "permalink": permalink, "comments": comments}


def _clean_comment(body, author):
    body = _clean(body) if "<" in (body or "") else (body or "").strip()
    if not body or body in ("[deleted]", "[removed]") or str(author) == "AutoModerator":
        return None
    return body


# --------------------------------------------------------------------------- #
#  Tier 1 — RSS/Atom feed (credential-free)
# --------------------------------------------------------------------------- #
def _fetch_via_rss(url: str, max_comments: int) -> dict:
    rss = normalize(url) + "/.rss"
    req = urllib.request.Request(rss, headers={
        "User-Agent": BROWSER_UA,
        "Accept": "application/atom+xml, application/xml, text/xml, */*"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
    if b"blocked by network security" in raw[:3000].lower():
        raise RuntimeError("blocked")
    root = ET.fromstring(raw)
    entries = root.findall("a:entry", _NS)
    if not entries:
        raise RuntimeError("Reddit feed had no entries.")

    def f(e, tag):
        x = e.find("a:" + tag, _NS)
        return x.text if x is not None else ""

    def author(e):
        a = e.find("a:author/a:name", _NS)
        name = (a.text if a is not None else "") or ""
        return name[3:] if name.startswith("/u/") else name  # store bare username

    # The post is the entry whose title is NOT "/u/<user> on …" (a comment).
    post_e = next((e for e in entries if not (f(e, "title") or "").startswith("/u/")),
                  entries[0])
    comment_es = [e for e in entries if e is not post_e]

    comments = []
    for e in comment_es:
        body = _clean(f(e, "content"))
        if body:
            comments.append({"body": body, "score": None, "author": author(e)})
        if len(comments) >= max_comments:
            break

    return _pack(f(post_e, "title"), _clean(f(post_e, "content")),
                 _subreddit(url), author(post_e), None, len(comment_es),
                 normalize(url), comments)


# --------------------------------------------------------------------------- #
#  Tier 2 — PRAW (official API, optional; adds score/num_comments)
# --------------------------------------------------------------------------- #
def _fetch_via_praw(url: str, max_comments: int) -> dict:
    import praw
    reddit = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                         client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                         user_agent=API_UA, check_for_async=False)
    reddit.read_only = True
    s = reddit.submission(url=normalize(url))
    comments = []
    try:
        s.comments.replace_more(limit=0)
        for c in s.comments:
            body = _clean_comment(getattr(c, "body", ""), getattr(c, "author", ""))
            if body:
                comments.append({"body": body, "score": int(getattr(c, "score", 0) or 0),
                                 "author": str(getattr(c, "author", ""))})
            if len(comments) >= max_comments:
                break
    except Exception:
        pass
    return _pack(s.title, s.selftext, str(s.subreddit), str(s.author), int(s.score),
                 s.num_comments, "https://www.reddit.com" + s.permalink, comments)


# --------------------------------------------------------------------------- #
#  Tier 3 — public .json (usually 403)
# --------------------------------------------------------------------------- #
def _fetch_via_json(url: str, max_comments: int) -> dict:
    jurl = normalize(url) + "/.json?raw_json=1&limit=50"
    req = urllib.request.Request(jurl, headers={"User-Agent": API_UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=12) as r:
        data = json.load(r)
    post = data[0]["data"]["children"][0]["data"]
    comments = []
    for c in data[1]["data"]["children"]:
        if c.get("kind") != "t1":
            continue
        cd = c.get("data", {})
        body = _clean_comment(cd.get("body"), cd.get("author"))
        if body:
            comments.append({"body": body, "score": int(cd.get("score", 0) or 0),
                             "author": cd.get("author", "")})
        if len(comments) >= max_comments:
            break
    return _pack(post.get("title"), post.get("selftext"), post.get("subreddit"),
                 post.get("author"), int(post.get("score", 0) or 0),
                 post.get("num_comments"),
                 "https://www.reddit.com" + post.get("permalink", ""), comments)


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def fetch_post(url: str, max_comments: int = 10) -> dict:
    normalize(url)  # validate early (ValueError on bad input)
    errors = []
    for name, fn, enabled in (
            ("rss", _fetch_via_rss, True),
            ("praw", _fetch_via_praw, _have_creds()),
            ("json", _fetch_via_json, True)):
        if not enabled:
            continue
        try:
            post = fn(url, max_comments)
            if post["title"] or post["selftext"] or post["comments"]:
                return post
            errors.append(f"{name}: empty")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Could not fetch this Reddit post (" + "; ".join(errors)
                       + "). It may be removed, private, or an image/link-only post.")

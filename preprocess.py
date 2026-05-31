"""
Faithful replication of the notebook's 23-step preprocessing pipeline
(Section 3 / Section 17). The saved Track B models — including the
transformers — were trained on, and run Reddit inference over, text that was:

  HTML/URL/newline stripped -> emoticons extracted -> contractions expanded ->
  non-alphanumeric removed (keep . _) -> repeated chars collapsed -> lowercased
  -> nltk word_tokenize + stopword removal (English minus negations) ->
  spaCy lemmatized (en_core_web_sm) -> emoticons re-appended.

Matching this at serve time keeps live predictions consistent with how the
models were trained. spaCy / nltk assets load lazily and are downloaded on
first use if missing (so the app still boots on a fresh Railway container).
"""
from __future__ import annotations

import re
import threading

# Exact emoticon regex from the notebook (Section 3, cell 18).
EMOTICON_RE = re.compile(
    r'(?:[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]'
    r'|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8]'
    r'|<3|</3|:\*)',
    re.IGNORECASE,
)

NEGATIONS = {'no', 'not', 'nor', 'never', 'neither', 'nobody', 'nothing',
             'nowhere', 'hardly', 'barely', 'scarcely'}

_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_NL_RE = re.compile(r'[\n\r]+')
_KEEP_RE = re.compile(r'[^a-zA-Z0-9\s._]')
_REPEAT_RE = re.compile(r'(.)\1{2,}')
_WS_RE = re.compile(r' +')

_lock = threading.Lock()
_assets = {"nlp": None, "stopwords": None, "tokenize": None, "ready": False}


def _ensure_assets():
    """Lazily load (and download if needed) spaCy + nltk assets, once."""
    if _assets["ready"]:
        return
    with _lock:
        if _assets["ready"]:
            return
        import nltk
        for pkg, path in [("punkt", "tokenizers/punkt"),
                          ("punkt_tab", "tokenizers/punkt_tab"),
                          ("stopwords", "corpora/stopwords"),
                          ("wordnet", "corpora/wordnet"),
                          ("omw-1.4", "corpora/omw-1.4")]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        _assets["nlp"] = nlp
        _assets["stopwords"] = set(stopwords.words("english")) - NEGATIONS
        _assets["tokenize"] = word_tokenize
        _assets["ready"] = True


def preprocess_text(text: str) -> str:
    """Transform one raw string exactly like the training/Reddit pipeline."""
    if not text:
        return ""
    import contractions

    s = str(text)
    s = _HTML_RE.sub("", s)
    s = _URL_RE.sub("", s)
    s = _NL_RE.sub(" ", s)
    emoticons = EMOTICON_RE.findall(s)
    s = EMOTICON_RE.sub(" ", s)
    s = contractions.fix(s)
    s = _KEEP_RE.sub("", s)
    s = _REPEAT_RE.sub(r"\1\1", s)
    s = _WS_RE.sub(" ", s.lower()).strip()

    _ensure_assets()
    tokens = [t for t in _assets["tokenize"](s)
              if t.lower() not in _assets["stopwords"]]
    doc = _assets["nlp"](" ".join(tokens))
    lemmas = [tok.lemma_ for tok in doc]
    return " ".join(lemmas + emoticons).strip()

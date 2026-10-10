"""Shared building blocks: corpora, URL features, grouping keys and classifiers.

Used by run_experiments.py (the experiments) and analysis.py (tables and figures).

Feature space: 50 URL features on three levels (URL, domain, directory/parameter).
The headline experiments use 49 of them (web_traffic_rank excluded, see README) and keep
the 20 with the highest Chi-squared score on the training partition.
"""
import html
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- config
SEED = 42
TEST_SIZE = 0.20          # 80% train / 20% test
TOP_K_FEATURES = 20       # Chi-squared top 20
CATBOOST_ITERATIONS = 300
UNRANKED = 1_000_001      # rank given to domains that are not in the top-1M list
ENSEMBLE_NAME = 'Soft-voting ensemble'
BASE_MODELS = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'CatBoost']
SPECIAL_CHARS = ['@', '#', '$', '%', '!', '.', '-', '+', '_', '=']
CHAR_NAMES = {'@': 'at', '#': 'hash', '$': 'dollar', '%': 'percent', '!': 'exclam',
              '.': 'dot', '-': 'hyphen', '+': 'plus', '_': 'underscore', '=': 'equals'}
SHORTENERS = {'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd', 'buff.ly',
              'adf.ly', 'bit.do', 'mcaf.ee', 'rebrand.ly', 'cutt.ly', 'shorturl.at',
              'tiny.cc', 'rb.gy', 'lnkd.in', 'j.mp', 'su.pr', 'cli.gs', 'v.gd'}


# ----------------------------------------------------------------------------- data
LABEL_ALIASES = {
    'benign': 'benign', 'legitimate': 'benign', 'legit': 'benign', 'good': 'benign',
    'alexa': 'benign', 'safe': 'benign', 'normal': 'benign',
    'phishing': 'phishing', 'phish': 'phishing', 'openphish': 'phishing',
    'spam': 'spam', 'webspam': 'spam',
}
URL_COLS = ['url', 'urls', 'text', 'link', 'domain']
LABEL_COLS = ['label', 'type', 'class', 'target', 'category', 'target_column', 'result']


def _label_from_filename(name):
    low = name.lower()
    for key in ['phish', 'spam', 'benign', 'legit', 'alexa']:
        if key in low:
            return LABEL_ALIASES[key]
    return None


def load_corpus(data_dir):
    """Read every CSV in data_dir into one frame with columns [url, label].

    Accepts either files with a URL column and a label column, or one-URL-per-line
    files whose class is given by the file name (benign / phishing / spam).
    """
    frames = []
    files = sorted(Path(data_dir).glob('*.csv'))
    if not files:
        sys.exit(f"No CSV files found in {data_dir}. See README.md.")
    for path in files:
        with open(path, encoding='utf-8', errors='replace') as fh:
            first = [c.strip().strip('"').lower() for c in fh.readline().lstrip('﻿').split(',')]
        has_header = any(c in URL_COLS + LABEL_COLS for c in first)
        if has_header:
            df = pd.read_csv(path, dtype=str, encoding_errors='replace')
            df.columns = [str(c).strip().lower() for c in df.columns]
            ucol = next((c for c in URL_COLS if c in df.columns), None)
            lcol = next((c for c in LABEL_COLS if c in df.columns), None)
        else:
            # one URL per line; URLs may contain commas, so do not parse as CSV
            with open(path, encoding='utf-8', errors='replace') as fh:
                lines = [ln.strip().lstrip('﻿').strip('"') for ln in fh]
            df = pd.DataFrame({'url': [ln for ln in lines if ln]})
            ucol, lcol = 'url', None
        if ucol is None:
            sys.exit(f"{path.name}: could not find a URL column (expected one of {URL_COLS}).")
        out = pd.DataFrame({'url': df[ucol].astype(str).str.strip()})
        if lcol is not None:
            out['label'] = df[lcol].astype(str).str.strip().str.lower()
        else:
            lab = _label_from_filename(path.name)
            if lab is None:
                sys.exit(f"{path.name}: no label column and the file name does not say "
                         f"benign / phishing / spam.")
            out['label'] = lab
        frames.append(out)
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus['url'].str.len() > 0].reset_index(drop=True)
    corpus['label'] = corpus['label'].map(lambda v: LABEL_ALIASES.get(v, v))
    return corpus


def load_iscx(data_dir=HERE / 'data' / 'iscx'):
    """Benign, phishing and spam URL lists of ISCX-URL-2016."""
    return load_corpus(data_dir)


def load_phiusiil(path=HERE / 'data' / 'phiusiil' / 'PhiUSIIL_Phishing_URL_Dataset.csv'):
    """PhiUSIIL URLs. In the source file label 1 is legitimate and label 0 is phishing."""
    df = pd.read_csv(path, usecols=['URL', 'label'], encoding_errors='replace')
    out = pd.DataFrame({'url': df['URL'].astype(str).str.strip(),
                        'label': df['label'].map({1: 'benign', 0: 'phishing'})})
    return out[out['url'].str.len() > 0].reset_index(drop=True)


# ----------------------------------------------------------------------------- URL parts
_SCHEME = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://')
_IPV4 = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)   # bundled suffix list, no network


def split_url(url):
    """Return (scheme, host, path, query) for a URL that may lack a scheme."""
    u = url.strip()
    has_scheme = bool(_SCHEME.match(u))
    try:
        p = urlsplit(u if has_scheme else 'http://' + u)
        host = (p.hostname or '').lower()
        return (p.scheme.lower() if has_scheme else ''), host, p.path, p.query
    except ValueError:
        rest = _SCHEME.sub('', u)
        hostpath, _, query = rest.partition('?')
        host, slash, path = hostpath.partition('/')
        return (u.split('://')[0].lower() if has_scheme else ''), host.lower(), slash + path, query


def registered_domain(host):
    """Registrable domain (public suffix plus one label); the host itself for IPs and unknown suffixes."""
    if not host or _IPV4.match(host):
        return host
    ext = _TLD(host)
    return f'{ext.domain}.{ext.suffix}' if ext.domain and ext.suffix else host


def public_suffix(host):
    if not host or _IPV4.match(host):
        return 'ip' if host else ''
    return _TLD(host).suffix or 'none'


def url_parts(urls):
    """Per-URL host, registered domain and public suffix (cached by host)."""
    hosts = [split_url(str(u))[1] for u in urls]
    uniq = set(hosts)
    dom = {h: registered_domain(h) for h in uniq}
    suf = {h: public_suffix(h) for h in uniq}
    return pd.DataFrame({'host': hosts, 'domain': [dom[h] for h in hosts],
                         'suffix': [suf[h] for h in hosts]})


class RankLookup:
    """Web-traffic rank of a host from a top-1M list (rank,domain CSV, optionally zipped)."""

    def __init__(self, path):
        path = Path(path)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=['rank', 'domain'])
        else:
            df = pd.read_csv(path, header=None, names=['rank', 'domain'])
        self.rank = dict(zip(df['domain'].astype(str), df['rank'].astype(int)))

    def __call__(self, host):
        h = host[4:] if host.startswith('www.') else host
        parts = h.split('.')
        for i in range(max(len(parts) - 1, 1)):
            r = self.rank.get('.'.join(parts[i:]))
            if r is not None:
                return r
        return UNRANKED


# ----------------------------------------------------------------------------- features
def extract_features(urls, rank_lookup=None, unescape=False):
    """50 features: 16 URL-level, 12 domain-level, 22 directory/parameter-level."""
    rows = []
    for url in urls:
        u = str(url).strip()
        if unescape:
            u = html.unescape(u)
        scheme, host, path, query = split_url(u)
        after_scheme = _SCHEME.sub('', u)
        bare = host[4:] if host.startswith('www.') else host
        f = {
            # --- URL level (16)
            'url_length': len(u),
            'url_depth': sum(1 for seg in path.split('/') if seg),
            'have_ip': int(bool(_IPV4.match(host))),
            'shortened_url': int(bare in SHORTENERS),
            'has_https': int(scheme == 'https'),
            'redirect': int('//' in after_scheme),
        }
        for c in SPECIAL_CHARS:
            f[f'url_count_{CHAR_NAMES[c]}'] = u.count(c)
        # --- domain level (12)
        for c in SPECIAL_CHARS:
            f[f'domain_count_{CHAR_NAMES[c]}'] = host.count(c)
        f['domain_length'] = len(host)
        f['web_traffic_rank'] = rank_lookup(host) if rank_lookup else UNRANKED
        # --- directory and parameter level (22)
        for c in SPECIAL_CHARS:
            f[f'dir_count_{CHAR_NAMES[c]}'] = path.count(c)
        f['dir_length'] = len(path)
        for c in SPECIAL_CHARS:
            f[f'param_count_{CHAR_NAMES[c]}'] = query.count(c)
        f['param_length'] = len(query)
        rows.append(f)
    X = pd.DataFrame(rows)
    assert X.shape[1] == 50, X.shape
    return X


def artefact_frame(urls, parts):
    """Collection artefacts only: public suffix, scheme, www prefix, HTML-escaped characters.

    None of these describes what a URL does; they describe where a list came from.
    """
    urls = pd.Series(list(urls), dtype=str)
    scheme = urls.str.extract(r'^([a-zA-Z][a-zA-Z0-9+.\-]*)://', expand=False).fillna('none').str.lower()
    return pd.DataFrame({
        'suffix': parts['suffix'].to_numpy(),
        'scheme': scheme.to_numpy(),
        'www': parts['host'].str.startswith('www.').astype(int).to_numpy(),
        'html_escape': urls.str.contains('&amp;', regex=False).astype(int).to_numpy(),
    })


# ----------------------------------------------------------------------------- models
def make_base_models(seed=SEED, params=None):
    """The four constituent classifiers. `params` optionally overrides hyperparameters per model."""
    p = params or {}
    return {
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed, **p.get('Logistic Regression', {}))),
        'Decision Tree': DecisionTreeClassifier(random_state=seed, **p.get('Decision Tree', {})),
        'Random Forest': RandomForestClassifier(
            **{'n_estimators': 100, 'random_state': seed, 'n_jobs': -1, **p.get('Random Forest', {})}),
        'CatBoost': CatBoostClassifier(
            **{'iterations': CATBOOST_ITERATIONS, 'random_seed': seed, 'verbose': 0,
               'allow_writing_files': False, **p.get('CatBoost', {})}),
    }


def make_artefact_model(seed=SEED):
    """Random forest on one-hot collection artefacts (rare suffixes pooled)."""
    encode = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), ['suffix', 'scheme'])],
        remainder='passthrough')
    return make_pipeline(encode, RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1))

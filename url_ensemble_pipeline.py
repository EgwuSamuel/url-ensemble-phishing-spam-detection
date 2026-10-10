"""Soft-voting ensemble for spam / phishing URL detection.

Pipeline:

  1. load the URL corpus (benign / phishing / spam)
  2. stratified 80/20 split
  3. 50 URL features on three levels (URL, domain, directory/parameter)
  4. Chi-squared selection of the 20 most discriminative features (fit on train only)
  5. Logistic Regression, Decision Tree, Random Forest, CatBoost
  6. soft-voting ensemble of the four
  7. accuracy, F1, ROC-AUC, fit / predict time, confusion matrices
  8. robustness checks: without the web-traffic-rank feature, and host-disjoint splits
     (no website in the test set appears in training)

Usage:
    python url_ensemble_pipeline.py
    python url_ensemble_pipeline.py --data DIR --out DIR --ranks FILE

Put the corpus in ./data and the rank list in ./resources (see README.md).
"""
import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- config
SEED = 42
TEST_SIZE = 0.20          # 80% train / 20% test
TOP_K_FEATURES = 20       # Chi-squared top 20 of 50
HOST_SPLITS = 5           # number of host-disjoint train/test splits (0 = off)
UNRANKED = 1_000_001      # rank given to domains that are not in the top-1M list
ENSEMBLE_NAME = 'Soft-Voting Ensemble (Proposed)'
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
        out['source_file'] = path.name
        frames.append(out)
        print(f"  loaded {len(out):>7,} rows from {path.name}")
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus['url'].str.len() > 0].reset_index(drop=True)
    corpus['label'] = corpus['label'].map(lambda v: LABEL_ALIASES.get(v, v))
    return corpus


# ----------------------------------------------------------------------------- features
_SCHEME = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://')
_IPV4 = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')


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


def extract_features(urls, rank_lookup):
    """50 features: 16 URL-level, 12 domain-level, 22 directory/parameter-level."""
    rows = []
    for url in urls:
        u = str(url).strip()
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


# ----------------------------------------------------------------------------- models
def make_models(seed=SEED):
    base = {
        'Logistic Regression': make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, random_state=seed)),
        'Decision Tree': DecisionTreeClassifier(random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        'CatBoost': CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False),
    }
    models = dict(base)
    models[ENSEMBLE_NAME] = VotingClassifier(
        estimators=[(k, clone(v)) for k, v in base.items()], voting='soft')
    return models


def score(model, X, y, n_classes):
    t0 = time.perf_counter()
    pred = model.predict(X)
    predict_time = time.perf_counter() - t0
    pred = np.asarray(pred).ravel().astype(int)
    proba = model.predict_proba(X)
    if n_classes == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class='ovr', average='macro')
    return pred, {'Predict Time (s)': round(predict_time, 3),
                  'Accuracy': round(accuracy_score(y, pred), 4),
                  'F1 Score': round(f1_score(y, pred, average='weighted'), 4),
                  'ROC-AUC': round(auc, 4)}


def run_experiment(name, train_df, test_df, X_train_all, X_test_all, encoder, out_dir,
                   drop=(), save_details=False):
    """Chi2 top-k on train, fit the five models, evaluate on test."""
    cols = [c for c in X_train_all.columns if c not in drop]
    y_train = encoder.transform(train_df['label'])
    y_test = encoder.transform(test_df['label'])
    n_classes = len(encoder.classes_)
    k = min(TOP_K_FEATURES, len(cols))
    selector = SelectKBest(chi2, k=k).fit(X_train_all[cols], y_train)
    chosen = [c for c, keep in zip(cols, selector.get_support()) if keep]
    Xtr, Xte = X_train_all[chosen].to_numpy(float), X_test_all[chosen].to_numpy(float)

    rows = []
    for model_name, model in make_models().items():
        t0 = time.perf_counter()
        model.fit(Xtr, y_train)
        fit_time = time.perf_counter() - t0
        pred, m = score(model, Xte, y_test, n_classes)
        rows.append({'Experiment': name, 'Model': model_name, 'Fit Time (s)': round(fit_time, 2), **m})
        print(f"    {model_name:<32} acc={m['Accuracy']:.4f} f1={m['F1 Score']:.4f} auc={m['ROC-AUC']:.4f}")
        if save_details:
            cm = confusion_matrix(y_test, pred, labels=range(n_classes))
            slug = re.sub(r'[^a-z0-9]+', '_', model_name.lower().replace('(proposed)', '')).strip('_')
            pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_) \
                .to_csv(out_dir / f'confusion_{slug}.csv')
            plot_confusion(cm, encoder.classes_, model_name, out_dir / f'confusion_{slug}.png')
            if model_name == ENSEMBLE_NAME:
                (out_dir / 'classification_report_ensemble.txt').write_text(
                    classification_report(y_test, pred, target_names=encoder.classes_, digits=4))
    if save_details:
        scores = pd.DataFrame({'feature': cols, 'chi2': selector.scores_,
                               'selected': selector.get_support()}) \
            .sort_values('chi2', ascending=False)
        scores.to_csv(out_dir / 'chi2_feature_scores.csv', index=False)
    return pd.DataFrame(rows), chosen


def plot_confusion(cm, labels, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--out', default=str(HERE / 'results'))
    ap.add_argument('--ranks', default=None, help='top-1M rank list (csv or zip)')
    ap.add_argument('--sample', type=int, default=0, help='use a random sample of N rows (testing)')
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading corpus")
    corpus = load_corpus(args.data)
    if args.sample:
        corpus = corpus.sample(n=min(args.sample, len(corpus)), random_state=SEED).reset_index(drop=True)
    counts = corpus['label'].value_counts().to_dict()
    print(f"  total {len(corpus):,} rows; classes: {counts}")
    dup = int(corpus['url'].duplicated().sum())
    print(f"  exact duplicate URLs: {dup:,}")

    print("2. Stratified 80/20 split")
    train_df, test_df = train_test_split(corpus, test_size=TEST_SIZE, random_state=SEED,
                                         stratify=corpus['label'])
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    overlap = int(test_df['url'].isin(set(train_df['url'])).sum())
    print(f"  train {len(train_df):,} / test {len(test_df):,}; test URLs also in train: {overlap:,}")

    rank_path = args.ranks or next(iter(sorted((HERE / 'resources').glob('tranco*'))), None)
    rank_lookup = RankLookup(rank_path) if rank_path else None
    print(f"  rank list: {Path(rank_path).name if rank_path else 'NONE (all URLs unranked)'}")

    print("3. Feature extraction (50 features)")
    encoder = LabelEncoder().fit(corpus['label'])
    X_train = extract_features(train_df['url'], rank_lookup)
    X_test = extract_features(test_df['url'], rank_lookup)

    print("4. Random split: five models on the Chi-squared top 20 features")
    main_table, chosen = run_experiment('Random split', train_df, test_df, X_train, X_test,
                                        encoder, out_dir, save_details=True)
    print("5. Random split without web_traffic_rank")
    norank_table, _ = run_experiment('Random split, without web_traffic_rank', train_df, test_df,
                                     X_train, X_test, encoder, out_dir, drop=('web_traffic_rank',))
    table = pd.concat([main_table, norank_table], ignore_index=True)
    table.to_csv(out_dir / 'metrics_all_experiments.csv', index=False)
    main_table.drop(columns='Experiment').to_csv(out_dir / 'table1_metrics.csv', index=False)

    host_info, agg = {}, None
    if HOST_SPLITS:
        # A random URL split puts URLs of the same website on both sides. In a host-disjoint
        # split no host in the test set appears in training, which is closer to meeting an
        # unseen website.
        print(f"6. {HOST_SPLITS} host-disjoint splits")
        hosts = corpus['url'].map(lambda u: split_url(u)[1])
        train_hosts = set(train_df['url'].map(lambda u: split_url(u)[1]))
        shared = float(test_df['url'].map(lambda u: split_url(u)[1]).isin(train_hosts).mean())
        print(f"  random split: {shared:.1%} of test URLs share a host with a training URL")
        X_all = extract_features(corpus['url'], rank_lookup)
        runs = []
        gss = GroupShuffleSplit(n_splits=HOST_SPLITS, test_size=TEST_SIZE, random_state=SEED)
        for n, (i_tr, i_te) in enumerate(gss.split(corpus, groups=hosts), 1):
            print(f"  split {n}: train {len(i_tr):,} / test {len(i_te):,}")
            for fs, drop in [('20 of 50 features', ()), ('without web_traffic_rank', ('web_traffic_rank',))]:
                r, _ = run_experiment(f'split {n}', corpus.iloc[i_tr], corpus.iloc[i_te],
                                      X_all.iloc[i_tr].reset_index(drop=True),
                                      X_all.iloc[i_te].reset_index(drop=True), encoder, out_dir, drop=drop)
                r.insert(1, 'Feature set', fs)
                r['Test rows'] = len(i_te)
                runs.append(r)
        runs = pd.concat(runs, ignore_index=True)
        runs.to_csv(out_dir / 'host_disjoint_splits.csv', index=False)
        agg = runs.groupby(['Feature set', 'Model'], sort=False)[['Accuracy', 'F1 Score', 'ROC-AUC']] \
            .agg(['mean', 'std']).round(4)
        agg.columns = [f'{a} {b}' for a, b in agg.columns]
        agg = agg.reset_index()
        agg.to_csv(out_dir / 'host_disjoint_summary.csv', index=False)
        host_info = {'unique_hosts': int(hosts.nunique()),
                     'hosts_per_class': corpus.assign(host=hosts).groupby('label')['host'].nunique().to_dict(),
                     'random_split_test_urls_sharing_host_with_train': round(shared, 4)}

    import sklearn, catboost
    summary = {
        'rows_total': len(corpus), 'class_counts': counts, 'duplicate_urls': dup,
        'train_rows': len(train_df), 'test_rows': len(test_df), 'test_urls_also_in_train': overlap,
        'selected_features': chosen, 'seed': SEED,
        'rank_list': Path(rank_path).name if rank_path else None,
        'host_overlap': host_info,
        'versions': {'python': sys.version.split()[0], 'scikit-learn': sklearn.__version__,
                     'catboost': catboost.__version__, 'pandas': pd.__version__},
    }
    (out_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))

    print("\n=== Random split ===")
    print(table.to_string(index=False))
    if agg is not None:
        print("\n=== Host-disjoint splits (mean and standard deviation) ===")
        print(agg.to_string(index=False))
    print(f"\nSelected features ({len(chosen)}): {chosen}")
    print(f"Outputs written to {out_dir}")


if __name__ == '__main__':
    main()

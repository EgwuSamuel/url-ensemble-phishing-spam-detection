"""Repeated-split experiments on ISCX-URL-2016 and PhiUSIIL.

Evaluation ladder:
  random        stratified 80/20 split of URLs
  host          80/20 split in which no host name is on both sides
  domain        80/20 split in which no registered domain is on both sides
  cross-corpus  train on one corpus, test on the other

Usage:
    python run_experiments.py all            # everything, in order (several hours on a CPU)
    python run_experiments.py iscx | tune | tuned | phiusiil | cross | cnn
    python run_experiments.py all --smoke    # tiny end-to-end check into results_smoke/

Every evaluated split appends one JSON line per model to <out>/raw/splits.jsonl, so an
interrupted run resumes where it stopped. analysis.py turns the raw files into tables
and figures.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline

import url_ensemble_pipeline as up
from char_models import CNN_NAME, NGRAM_NAME, fit_predict_cnn, make_ngram_model

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
ARTEFACT_NAME = 'Artefact-only RF'
N_SPLITS = {'iscx': 30, 'phiusiil': 20}          # per protocol, headline experiment
N_SIDE = {'iscx': 10, 'phiusiil': 5}             # per protocol, sensitivity experiments
N_CNN = {'iscx': 5, 'phiusiil': 3}               # per protocol, character CNN


# ----------------------------------------------------------------------------- context
class Corpus:
    """A corpus prepared once: labels, feature matrix, grouping keys, artefact columns."""

    def __init__(self, name, df, rank_lookup=None, unescape=False):
        self.name = name
        self.df = df.reset_index(drop=True)
        self.urls = self.df['url'].to_numpy()
        self.classes = sorted(self.df['label'].unique())
        self.y = self.df['label'].map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        self.parts = up.url_parts(self.urls)
        self.groups = {'host': self.parts['host'].to_numpy(), 'domain': self.parts['domain'].to_numpy()}
        self.X = up.extract_features(self.urls, rank_lookup, unescape=unescape)
        self.A = up.artefact_frame(self.urls, self.parts)

    def subset(self, mask, name=None, relabel=None):
        df = self.df[mask].copy()
        if relabel:
            df['label'] = df['label'].map(relabel)
        c = Corpus.__new__(Corpus)
        c.name = name or self.name
        c.df = df.reset_index(drop=True)
        c.urls = c.df['url'].to_numpy()
        c.classes = sorted(c.df['label'].unique())
        c.y = c.df['label'].map({k: i for i, k in enumerate(c.classes)}).to_numpy()
        c.parts = self.parts[mask].reset_index(drop=True)
        c.groups = {k: v[mask] for k, v in self.groups.items()}
        c.X = self.X[mask].reset_index(drop=True)
        c.A = self.A[mask].reset_index(drop=True)
        return c


def make_splits(corpus, protocol, n):
    dummy = np.zeros(len(corpus.y))
    if protocol == 'random':
        gen = StratifiedShuffleSplit(n_splits=n, test_size=up.TEST_SIZE, random_state=up.SEED).split(dummy, corpus.y)
    else:
        g = corpus.groups[protocol]
        gen = GroupShuffleSplit(n_splits=n, test_size=up.TEST_SIZE, random_state=up.SEED).split(dummy, corpus.y, g)
    for i, (tr, te) in enumerate(gen):
        if protocol != 'random':
            assert not set(corpus.groups[protocol][tr]) & set(corpus.groups[protocol][te]), 'group leak'
        yield i, tr, te


# ----------------------------------------------------------------------------- metrics
def metrics(y, proba, classes):
    pred = proba.argmax(1)
    labels = list(range(len(classes)))
    out = {'accuracy': accuracy_score(y, pred),
           'f1_macro': f1_score(y, pred, average='macro', labels=labels, zero_division=0),
           'f1_weighted': f1_score(y, pred, average='weighted', labels=labels, zero_division=0)}
    try:
        out['roc_auc'] = (roc_auc_score(y, proba[:, 1]) if len(classes) == 2
                          else roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=labels))
    except ValueError:            # a class is missing from this test partition
        out['roc_auc'] = None
    p, r, _, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    for c, pc, rc in zip(classes, p, r):
        out[f'precision_{c}'], out[f'recall_{c}'] = float(pc), float(rc)
    return out, pred


class Sink:
    """Append-only JSON-lines output with resume support."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir) / 'raw'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.done = set()
        f = self.dir / 'splits.jsonl'
        if f.exists():
            for line in f.read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                self.done.add((r['stage'], r['corpus'], r['experiment'], r['protocol'], r['split']))

    def write(self, name, rows):
        with open(self.dir / f'{name}.jsonl', 'a', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')


def record(sink, key, model, y, proba, classes, n_train, fit_time, pred_time, rows, cms):
    m, pred = metrics(y, proba, classes)
    rows.append({**key, 'model': model, 'n_train': int(n_train), 'n_test': int(len(y)),
                 'fit_time': round(fit_time, 3), 'predict_time': round(pred_time, 4), **m})
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            cms.append({**key, 'model': model, 'true': t, 'pred': p, 'count': int(cm[i, j])})
    return pred


# ----------------------------------------------------------------------------- one split
def run_split(sink, corpus, key, tr, te, cols, params=None, extras=True, save_curves=False, save_errors=False):
    """Fit and score every model on one train/test partition; returns the ensemble accuracy."""
    y_tr, y_te, classes = corpus.y[tr], corpus.y[te], corpus.classes
    rows, cms = [], []
    suffix = ' (tuned)' if params else ''

    sel = SelectKBest(chi2, k=min(up.TOP_K_FEATURES, len(cols))).fit(corpus.X.iloc[tr][cols], y_tr)
    chosen = [c for c, keep in zip(cols, sel.get_support()) if keep]
    Xtr = corpus.X.iloc[tr][chosen].to_numpy(float)
    Xte = corpus.X.iloc[te][chosen].to_numpy(float)

    probas, fit_total, pred_total = [], 0.0, 0.0
    for name, model in up.make_base_models(params=params).items():
        t0 = time.perf_counter(); model.fit(Xtr, y_tr); ft = time.perf_counter() - t0
        t0 = time.perf_counter(); pr = model.predict_proba(Xte); pt = time.perf_counter() - t0
        probas.append(pr); fit_total += ft; pred_total += pt
        record(sink, key, name + suffix, y_te, pr, classes, len(tr), ft, pt, rows, cms)
    ens = np.mean(probas, axis=0)                      # soft vote: unweighted mean of probabilities
    ens_pred = record(sink, key, up.ENSEMBLE_NAME + suffix, y_te, ens, classes, len(tr),
                      fit_total, pred_total, rows, cms)

    if extras:
        for name, model, Atr, Ate in [
                (NGRAM_NAME, make_ngram_model(), corpus.urls[tr], corpus.urls[te]),
                (ARTEFACT_NAME, up.make_artefact_model(), corpus.A.iloc[tr], corpus.A.iloc[te])]:
            t0 = time.perf_counter(); model.fit(Atr, y_tr); ft = time.perf_counter() - t0
            t0 = time.perf_counter(); pr = model.predict_proba(Ate); pt = time.perf_counter() - t0
            record(sink, key, name, y_te, pr, classes, len(tr), ft, pt, rows, cms)

    sink.write('selected', [{**key, 'features': chosen}])
    if save_curves:
        curves = []
        for i, c in enumerate(classes if len(classes) > 2 else classes[1:]):
            k = i if len(classes) > 2 else 1
            fpr, tpr, _ = roc_curve((y_te == k).astype(int), ens[:, k])
            idx = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
            curves.append({**key, 'model': up.ENSEMBLE_NAME, 'class': c,
                           'fpr': [round(float(v), 5) for v in fpr[idx]],
                           'tpr': [round(float(v), 5) for v in tpr[idx]]})
        sink.write('roc', curves)
    if save_errors:
        t = pd.DataFrame({'domain': corpus.parts['domain'].to_numpy()[te], 'suffix': corpus.parts['suffix'].to_numpy()[te],
                          'true': [classes[i] for i in y_te], 'pred': [classes[i] for i in ens_pred]})
        tot = t.groupby(['domain', 'suffix', 'true']).size().rename('n').reset_index()
        err = t[t['true'] != t['pred']].groupby(['domain', 'suffix', 'true', 'pred']).size().rename('errors').reset_index()
        sink.write('errors', [{**key, **r} for r in err.to_dict('records')])
        sink.write('test_domains', [{**key, **r} for r in tot.to_dict('records')])
    sink.write('confusions', cms)
    sink.write('splits', rows)            # written last: marks the split as complete
    return rows[4]['accuracy']


def run_protocols(sink, corpus, experiment, plan, cols, params=None, extras=True, detail=False, stage='main'):
    for protocol, n in plan.items():
        for i, tr, te in make_splits(corpus, protocol, n):
            key = {'stage': stage, 'corpus': corpus.name, 'experiment': experiment, 'protocol': protocol, 'split': i}
            if tuple(key.values()) in sink.done:
                continue
            t0 = time.time()
            acc = run_split(sink, corpus, key, tr, te, cols, params=params,
                            extras=extras and protocol != 'host',
                            save_curves=detail and i == 0, save_errors=detail and protocol != 'random')
            print(f"  {corpus.name:<16} {experiment:<16} {protocol:<7} split {i:>2}  "
                  f"test {len(te):>6,}  ensemble acc {acc:.4f}  ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------- tuning
GRIDS = {
    'Logistic Regression': {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
    'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 5, 10]},
    'Random Forest': {'n_estimators': [200, 400], 'max_depth': [None, 20],
                      'min_samples_leaf': [1, 2, 5], 'max_features': ['sqrt', 0.5]},
    'CatBoost': {'depth': [4, 6, 8], 'learning_rate': [0.03, 0.1, 0.3], 'l2_leaf_reg': [1, 3, 10],
                 'iterations': [300, 600]},
}
N_CANDIDATES = {'Logistic Regression': 5, 'Decision Tree': 8, 'Random Forest': 8, 'CatBoost': 6}


def tune(corpus, cols, max_rows=20_000):
    """Random search with grouped 3-fold CV on the training part of the first domain-disjoint split.

    The library defaults are always among the candidates. Feature selection sits inside the CV.
    """
    _, tr, _ = next(make_splits(corpus, 'domain', 1))
    rng = np.random.default_rng(up.SEED)
    if len(tr) > max_rows:
        tr = rng.choice(tr, max_rows, replace=False)
    X, y, g = corpus.X.iloc[tr][cols].to_numpy(float), corpus.y[tr], corpus.groups['domain'][tr]
    folds = list(GroupKFold(n_splits=3).split(X, y, g))
    best, log = {}, []
    for name, grid in GRIDS.items():
        keys = list(grid)
        combos = [dict(zip(keys, vals)) for vals in np.array(np.meshgrid(*[np.arange(len(grid[k])) for k in keys]))
                  .T.reshape(-1, len(keys)).tolist()]
        combos = [{k: grid[k][i] for k, i in c.items()} for c in combos]
        picks = [combos[i] for i in rng.choice(len(combos), min(N_CANDIDATES[name], len(combos)), replace=False)]
        scored = []
        for cand in [{}] + picks:                       # {} = library defaults
            accs = []
            for ftr, fte in folds:
                model = make_pipeline(SelectKBest(chi2, k=min(up.TOP_K_FEATURES, X.shape[1])),
                                      up.make_base_models(params={name: cand})[name])
                model.fit(X[ftr], y[ftr])
                accs.append(accuracy_score(y[fte], np.asarray(model.predict(X[fte])).ravel().astype(int)))
            scored.append((float(np.mean(accs)), cand))
            log.append({'corpus': corpus.name, 'model': name, 'params': cand, 'cv_accuracy': round(float(np.mean(accs)), 4)})
            print(f"  tune {corpus.name} {name:<20} {np.mean(accs):.4f}  {cand or 'defaults'}")
        best[name] = max(scored, key=lambda s: s[0])[1]
    return best, log


# ----------------------------------------------------------------------------- stages
def headline_cols(corpus):
    return [c for c in corpus.X.columns if c != 'web_traffic_rank']


def load(name, smoke, rank_lookup=None, unescape=False):
    df = up.load_iscx() if name == 'iscx' else up.load_phiusiil()
    if smoke:
        df = df.groupby('label', group_keys=False).sample(n=min(1500, df['label'].value_counts().min()),
                                                           random_state=up.SEED)
    label = 'ISCX-URL-2016' if name == 'iscx' else 'PhiUSIIL'
    return Corpus(label, df, rank_lookup, unescape)


def n(k, table, smoke):
    return 2 if smoke else table[k]


def stage_iscx(sink, smoke):
    rank_path = next(iter(sorted((HERE / 'resources').glob('tranco*'))), None)
    c = load('iscx', smoke, up.RankLookup(rank_path) if rank_path else None)
    cols = headline_cols(c)
    big, side = n('iscx', N_SPLITS, smoke), n('iscx', N_SIDE, smoke)
    run_protocols(sink, c, 'headline', {'random': big, 'domain': big, 'host': big}, cols, detail=True)
    if rank_path:
        run_protocols(sink, c, 'with_rank', {'random': side, 'domain': side}, list(c.X.columns), extras=False)
    b = c.subset(np.ones(len(c.y), bool), relabel={'benign': 'benign', 'phishing': 'malicious', 'spam': 'malicious'})
    run_protocols(sink, b, 'benign_vs_malicious', {'random': side, 'domain': side}, cols, extras=False)
    bp = c.subset((c.df['label'] != 'spam').to_numpy())
    run_protocols(sink, bp, 'benign_vs_phishing', {'random': side, 'domain': side}, cols)
    u = load('iscx', smoke, unescape=True)
    run_protocols(sink, u, 'unescaped', {'random': side, 'domain': side}, headline_cols(u), extras=False)


def stage_phiusiil(sink, smoke):
    c = load('phiusiil', smoke)
    big = n('phiusiil', N_SPLITS, smoke)
    run_protocols(sink, c, 'headline', {'random': big, 'domain': big}, headline_cols(c), detail=True)


def stage_tune(sink, smoke):
    out, logs = {}, []
    for name in ['iscx', 'phiusiil']:
        c = load(name, smoke)
        best, log = tune(c, headline_cols(c), max_rows=2000 if smoke else 20_000)
        out[c.name] = best
        logs += log
    (sink.dir / 'tuned_params.json').write_text(json.dumps(out, indent=2))
    (sink.dir / 'tuning_log.json').write_text(json.dumps(logs, indent=1))


def stage_tuned(sink, smoke):
    params = json.loads((sink.dir / 'tuned_params.json').read_text())
    for name in ['iscx', 'phiusiil']:
        c = load(name, smoke)
        side = n(name, N_SIDE, smoke)
        run_protocols(sink, c, 'tuned', {'random': side, 'domain': side}, headline_cols(c),
                      params=params[c.name], extras=False)


def stage_cnn(sink, smoke):
    for name in ['iscx', 'phiusiil']:
        c = load(name, smoke)
        k = n(name, N_CNN, smoke)
        for protocol in ['random', 'domain']:
            for i, tr, te in make_splits(c, protocol, n(name, N_SPLITS, smoke)):
                if i >= k:
                    break
                key = {'stage': 'cnn', 'corpus': c.name, 'experiment': 'headline', 'protocol': protocol, 'split': i}
                if tuple(key.values()) in sink.done:
                    continue
                t0 = time.perf_counter()
                pr = fit_predict_cnn(c.urls[tr], c.y[tr], c.urls[te], len(c.classes), epochs=1 if smoke else 2)
                rows, cms = [], []
                record(sink, key, CNN_NAME, c.y[te], pr, c.classes, len(tr), time.perf_counter() - t0, 0.0, rows, cms)
                sink.write('confusions', cms); sink.write('splits', rows)
                print(f"  {c.name:<16} cnn {protocol:<7} split {i}  acc {rows[0]['accuracy']:.4f}  "
                      f"({time.perf_counter() - t0:.0f}s)")


def stage_cross(sink, smoke):
    """Train on all of one corpus, test on all of the other (benign vs phishing)."""
    a = load('iscx', smoke)
    a = a.subset((a.df['label'] != 'spam').to_numpy())
    b = load('phiusiil', smoke)
    for src, dst in [(a, b), (b, a)]:
        name = f'{src.name} -> {dst.name}'
        key = {'stage': 'cross', 'corpus': name, 'experiment': 'headline', 'protocol': 'cross-corpus', 'split': 0}
        if tuple(key.values()) in sink.done:
            continue
        cols = headline_cols(src)
        both = Corpus.__new__(Corpus)
        both.name, both.classes = name, src.classes
        both.urls = np.concatenate([src.urls, dst.urls])
        both.y = np.concatenate([src.y, dst.y])
        both.X = pd.concat([src.X, dst.X], ignore_index=True)
        both.A = pd.concat([src.A, dst.A], ignore_index=True)
        both.parts = pd.concat([src.parts, dst.parts], ignore_index=True)
        tr, te = np.arange(len(src.y)), np.arange(len(src.y), len(src.y) + len(dst.y))
        t0 = time.time()
        acc = run_split(sink, both, key, tr, te, cols)
        pr = fit_predict_cnn(both.urls[tr], both.y[tr], both.urls[te], 2, epochs=1 if smoke else 2)
        rows, cms = [], []
        record(sink, {**key, 'stage': 'cross-cnn'}, CNN_NAME, both.y[te], pr, both.classes, len(tr), 0.0, 0.0, rows, cms)
        sink.write('confusions', cms); sink.write('splits', rows)
        print(f"  {name}: ensemble acc {acc:.4f}, cnn acc {rows[0]['accuracy']:.4f} ({time.time() - t0:.0f}s)")


STAGES = {'iscx': stage_iscx, 'tune': stage_tune, 'tuned': stage_tuned, 'phiusiil': stage_phiusiil,
          'cross': stage_cross, 'cnn': stage_cnn}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=list(STAGES) + ['all'])
    ap.add_argument('--smoke', action='store_true', help='tiny subsample and 2 splits, into results_smoke/')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE / ('results_smoke' if args.smoke else 'results')
    sink = Sink(out)
    for name in (list(STAGES) if args.stage == 'all' else [args.stage]):
        print(f"== stage {name}")
        t0 = time.time()
        STAGES[name](sink, args.smoke)
        print(f"== stage {name} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()

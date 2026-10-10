"""Tables and figures from the raw experiment output.

    python analysis.py                # reads results/raw, writes results/*.csv and results/figures/*.png
    python analysis.py --dir results_smoke

Intervals use the corrected resampled variance of Nadeau and Bengio (2003): repeated
train/test splits overlap, so the naive standard error is too small. For n splits with
test/train size ratio r, the variance of the mean is taken as (1/n + r) * s^2.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import url_ensemble_pipeline as up
from char_models import CNN_NAME, NGRAM_NAME

HERE = Path(__file__).resolve().parent
ARTEFACT_NAME = 'Artefact-only RF'
ENS = up.ENSEMBLE_NAME
FEATURE_MODELS = up.BASE_MODELS + [ENS]
METRICS = ['accuracy', 'f1_macro', 'roc_auc']
PROTOCOL_LABEL = {'random': 'Random split', 'host': 'Host-disjoint', 'domain': 'Domain-disjoint',
                  'cross-corpus': 'Cross-corpus'}
# validated categorical palette (blue, orange, aqua); see README
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
PROTOCOL_STYLE = {'random': (BLUE, 'o'), 'host': (AQUA, '^'), 'domain': (ORANGE, 's')}
CLASS_COLOR = {'benign': BLUE, 'phishing': ORANGE, 'spam': AQUA, 'malicious': ORANGE}


# ----------------------------------------------------------------------------- statistics
def corrected_interval(x, ratio, level=0.95):
    """Mean, SD and corrected-resampled confidence interval of split scores."""
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, ci_low=np.nan, ci_high=np.nan, se=np.nan)
    m = x.mean()
    if n == 1:
        return dict(n=1, mean=m, sd=np.nan, ci_low=np.nan, ci_high=np.nan, se=np.nan)
    sd = x.std(ddof=1)
    se = np.sqrt((1 / n + ratio) * sd ** 2)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return dict(n=n, mean=m, sd=sd, ci_low=m - h, ci_high=m + h, se=se)


def corrected_paired_test(a, b, ratio):
    """Corrected resampled t-test and Wilcoxon signed-rank test on paired split scores."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    out = {'n': n, 'mean_diff': d.mean()}
    sd = d.std(ddof=1) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t = d.mean() / np.sqrt((1 / n + ratio) * sd ** 2)
        out['p_corrected_t'] = float(2 * stats.t.sf(abs(t), n - 1))
        h = stats.t.ppf(0.975, n - 1) * np.sqrt((1 / n + ratio) * sd ** 2)
        out['diff_ci_low'], out['diff_ci_high'] = d.mean() - h, d.mean() + h
    else:
        out['p_corrected_t'] = 1.0 if d.mean() == 0 else np.nan
        out['diff_ci_low'] = out['diff_ci_high'] = d.mean()
    try:
        out['p_wilcoxon'] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and n > 5 else np.nan
    except ValueError:
        out['p_wilcoxon'] = np.nan
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj, running, m = np.full(len(p), np.nan), 0.0, int(np.sum(~np.isnan(p)))
    for rank, i in enumerate(order):
        if np.isnan(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def wilson(p, n, z=1.96):
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (c - h) / d, (c + h) / d


# ----------------------------------------------------------------------------- loading
def read_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:      # a line still being written by a running experiment
            continue
    df = pd.DataFrame(rows)
    # a split that was interrupted and re-run leaves earlier partial rows behind: keep the latest
    keys = [c for c in ['stage', 'corpus', 'experiment', 'protocol', 'split', 'model', 'class', 'true', 'pred',
                        'domain', 'suffix'] if c in df.columns]
    return df.drop_duplicates(subset=keys, keep='last').reset_index(drop=True) if keys else df


def ratio_of(g):
    return float((g['n_test'] / g['n_train']).mean())


# ----------------------------------------------------------------------------- tables
def summary_table(sp):
    rows = []
    for (corpus, exp, proto, model), g in sp.groupby(['corpus', 'experiment', 'protocol', 'model'], sort=False):
        row = {'corpus': corpus, 'experiment': exp, 'protocol': proto, 'model': model, 'splits': len(g),
               'mean_test_rows': int(g['n_test'].mean())}
        for m in METRICS:
            ci = corrected_interval(g[m], ratio_of(g))
            row.update({f'{m}_mean': ci['mean'], f'{m}_sd': ci['sd'], f'{m}_ci_low': ci['ci_low'],
                        f'{m}_ci_high': ci['ci_high'], f'{m}_se': ci['se']})
        row['fit_time_median'] = g['fit_time'].median()
        row['predict_time_median'] = g['predict_time'].median()
        rows.append(row)
    return pd.DataFrame(rows)


def inflation_table(summ):
    rows = []
    for (corpus, exp, model), g in summ.groupby(['corpus', 'experiment', 'model'], sort=False):
        g = g.set_index('protocol')
        if 'random' not in g.index:
            continue
        for strict in ['host', 'domain']:
            if strict not in g.index:
                continue
            r, s = g.loc['random'], g.loc[strict]
            for m in ['accuracy', 'f1_macro']:
                diff = r[f'{m}_mean'] - s[f'{m}_mean']
                se = np.sqrt(np.nansum([r[f'{m}_se'] ** 2, s[f'{m}_se'] ** 2]))
                df = max(min(r['splits'], s['splits']) - 1, 1)
                h = stats.t.ppf(0.975, df) * se
                rows.append({'corpus': corpus, 'experiment': exp, 'model': model, 'against': strict, 'metric': m,
                             'random_mean': r[f'{m}_mean'], 'strict_mean': s[f'{m}_mean'], 'inflation': diff,
                             'ci_low': diff - h, 'ci_high': diff + h})
    return pd.DataFrame(rows)


def paired_table(sp, models):
    rows = []
    hl = sp[sp['experiment'] == 'headline']
    for (corpus, proto), g in hl.groupby(['corpus', 'protocol'], sort=False):
        wide = g.pivot_table(index='split', columns='model', values='accuracy')
        block = []
        for a, b in itertools.combinations([m for m in models if m in wide.columns], 2):
            pair = wide[[a, b]].dropna()
            if len(pair) < 2:
                continue
            t = corrected_paired_test(pair[a], pair[b], ratio_of(g))
            block.append({'corpus': corpus, 'protocol': proto, 'model_a': a, 'model_b': b, **t})
        if block:
            b = pd.DataFrame(block)
            b['p_corrected_t_holm'] = holm(b['p_corrected_t'])
            b['p_wilcoxon_holm'] = holm(b['p_wilcoxon'])
            rows.append(b)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_against_headline(sp, experiment, strip=''):
    """Accuracy of an alternative experiment minus the headline run on the same splits."""
    rows = []
    alt = sp[sp['experiment'] == experiment].copy()
    if strip:      # an empty pattern must not reach str.replace
        alt['model'] = alt['model'].map(lambda m: m.replace(strip, ''))
    hl = sp[sp['experiment'] == 'headline']
    for (corpus, proto, model), g in alt.groupby(['corpus', 'protocol', 'model'], sort=False):
        base = hl[(hl['corpus'] == corpus) & (hl['protocol'] == proto) & (hl['model'] == model)].set_index('split')
        both = g.set_index('split').join(base[['accuracy']], rsuffix='_headline', how='inner')
        if len(both) < 2:
            continue
        t = corrected_paired_test(both['accuracy'], both['accuracy_headline'], ratio_of(g))
        rows.append({'corpus': corpus, 'protocol': proto, 'model': model, 'experiment': experiment,
                     'alternative_mean': both['accuracy'].mean(), 'headline_mean': both['accuracy_headline'].mean(), **t})
    return pd.DataFrame(rows)


def per_class_table(sp, models):
    rows = []
    hl = sp[(sp['experiment'] == 'headline') & sp['model'].isin(models)]
    cols = [c for c in sp.columns if c.startswith(('precision_', 'recall_'))]
    for (corpus, proto, model), g in hl.groupby(['corpus', 'protocol', 'model'], sort=False):
        for c in cols:
            if g[c].notna().sum() == 0:
                continue
            ci = corrected_interval(g[c], ratio_of(g))
            kind, cls = c.split('_', 1)
            rows.append({'corpus': corpus, 'protocol': proto, 'model': model, 'class': cls, 'measure': kind,
                         'mean': ci['mean'], 'sd': ci['sd'], 'ci_low': ci['ci_low'], 'ci_high': ci['ci_high']})
    return pd.DataFrame(rows)


def cross_table(sp):
    rows = []
    for _, r in sp[sp['protocol'] == 'cross-corpus'].iterrows():
        lo, hi = wilson(r['accuracy'], r['n_test'])
        rows.append({'direction': r['corpus'], 'model': r['model'], 'n_train': r['n_train'], 'n_test': r['n_test'],
                     'accuracy': r['accuracy'], 'ci_low': lo, 'ci_high': hi, 'f1_macro': r['f1_macro'],
                     'roc_auc': r['roc_auc'], 'recall_benign': r.get('recall_benign'),
                     'recall_phishing': r.get('recall_phishing')})
    return pd.DataFrame(rows)


def corpus_audit():
    rows, curves = [], {}
    for name, df in [('ISCX-URL-2016', up.load_iscx()), ('PhiUSIIL', up.load_phiusiil())]:
        parts = up.url_parts(df['url'])
        art = up.artefact_frame(df['url'], parts)
        full = pd.concat([df.reset_index(drop=True), parts, art[['scheme', 'www', 'html_escape']]], axis=1)
        for cls, g in full.groupby('label'):
            vc = g['domain'].value_counts()
            top_suffix = g['suffix'].value_counts(normalize=True).head(3)
            rows.append({'corpus': name, 'class': cls, 'urls': len(g), 'duplicate_urls': int(g['url'].duplicated().sum()),
                         'hosts': g['host'].nunique(), 'registered_domains': g['domain'].nunique(),
                         'public_suffixes': g['suffix'].nunique(),
                         'top10_domain_share': vc.head(10).sum() / len(g),
                         'https_share': (g['scheme'] == 'https').mean(), 'no_scheme_share': (g['scheme'] == 'none').mean(),
                         'www_share': g['www'].mean(), 'html_escape_share': g['html_escape'].mean(),
                         'top_suffixes': '; '.join(f'{s} {v:.1%}' for s, v in top_suffix.items())})
            curves[(name, cls)] = np.cumsum(vc.to_numpy()) / len(g)
        shared = full.groupby('domain')['label'].nunique()
        rows.append({'corpus': name, 'class': 'ALL', 'urls': len(full), 'duplicate_urls': int(full['url'].duplicated().sum()),
                     'hosts': full['host'].nunique(), 'registered_domains': full['domain'].nunique(),
                     'public_suffixes': full['suffix'].nunique(),
                     'domains_in_more_than_one_class': int((shared > 1).sum())})
    return pd.DataFrame(rows), curves


# ----------------------------------------------------------------------------- figures
def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'legend.frameon': False, 'savefig.dpi': 300, 'figure.dpi': 120,
        'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
    return plt


def fig_pipeline(out):
    plt = style()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)

    def box(x, y, w, h, text, fc='#ffffff', ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08', fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.2))

    ax.text(0.05, 4.35, 'Detection pipeline (identical under every protocol)', fontsize=9, fontweight='bold')
    xs = [0.05, 2.1, 4.15, 6.2, 8.25]
    labels = ['URL string', '49 lexical\nfeatures', 'Chi-squared\ntop 20\n(training data only)',
              'LR, Decision Tree,\nRandom Forest,\nCatBoost', 'Soft vote\n(mean class\nprobability)']
    for x, t in zip(xs, labels):
        box(x, 2.95, 1.7, 1.1, t, fc='#eef4fc')
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 1.7, 3.5, b, 3.5)
    ax.text(0.05, 2.35, 'Evaluation ladder (each step removes one source of optimism)', fontsize=9, fontweight='bold')
    steps = [('1  Random split', 'one website may be\non both sides', BLUE), ('2  Host-disjoint', 'no host name on\nboth sides', AQUA),
             ('3  Domain-disjoint', 'no registered domain\non both sides', ORANGE), ('4  Cross-corpus', 'train on one corpus,\ntest on the other', INK2)]
    for i, (head, sub, col) in enumerate(steps):
        x = 0.05 + i * 2.5
        box(x, 0.55, 2.2, 1.35, '', ec=col)
        ax.text(x + 1.1, 1.55, head, ha='center', va='center', fontsize=7.6, fontweight='bold')
        ax.text(x + 1.1, 1.0, sub, ha='center', va='center', fontsize=6.8, color=INK2)
        if i < 3:
            arrow(x + 2.2, 1.22, x + 2.5, 1.22)
    ax.text(5.0, 0.15, 'stricter  →', ha='center', fontsize=8, color=INK2)
    fig.savefig(out / 'fig1_pipeline_and_ladder.png', bbox_inches='tight'); plt.close(fig)


def fig_concentration(curves, out):
    plt = style()
    corpora = sorted({k[0] for k in curves})
    fig, axes = plt.subplots(1, len(corpora), figsize=(7.2, 3.0), sharey=True)
    for ax, corpus in zip(np.atleast_1d(axes), corpora):
        for (c, cls), y in curves.items():
            if c != corpus:
                continue
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=CLASS_COLOR[cls], lw=2, label=f'{cls} ({len(y):,} domains)')
        ax.set_xscale('log'); ax.set_title(corpus); ax.set_xlabel('Registered domains, largest first (log scale)')
        ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.legend(loc='lower right' if corpus.startswith('ISCX') else 'upper left', fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel('Cumulative share of the class’s URLs')
    fig.tight_layout(); fig.savefig(out / 'fig2_domain_concentration.png', bbox_inches='tight'); plt.close(fig)


def fig_ladder(summ, cross, out):
    plt = style()
    models = FEATURE_MODELS + [NGRAM_NAME, CNN_NAME, ARTEFACT_NAME]
    hl = summ[summ['experiment'] == 'headline']
    corpora = [c for c in ['ISCX-URL-2016', 'PhiUSIIL'] if c in set(hl['corpus'])]
    fig, axes = plt.subplots(1, len(corpora), figsize=(7.2, 3.9), sharey=True)
    for ax, corpus in zip(np.atleast_1d(axes), corpora):
        g = hl[hl['corpus'] == corpus]
        present = [m for m in models if m in set(g['model'])]
        ypos = {m: i for i, m in enumerate(present[::-1])}
        offs = {'random': 0.22, 'host': 0.0, 'domain': -0.22}
        for proto, (col, mk) in PROTOCOL_STYLE.items():
            s = g[g['protocol'] == proto]
            if s.empty:
                continue
            y = [ypos[m] + offs[proto] for m in s['model']]
            lo = (s['accuracy_mean'] - s['accuracy_ci_low']).fillna(0).clip(lower=0)
            hi = (s['accuracy_ci_high'].clip(upper=1.0) - s['accuracy_mean']).fillna(0).clip(lower=0)
            ax.errorbar(s['accuracy_mean'], y, xerr=[lo, hi], fmt=mk, color=col, ms=5.5, lw=1.4, capsize=2.5,
                        mec='white', mew=0.8, label=PROTOCOL_LABEL[proto])
        if cross is not None and not cross.empty:
            c = cross[cross['direction'].str.endswith('-> ' + corpus)]
            c = c[c['model'].isin(present)]
            ax.scatter(c['accuracy'], [ypos[m] - 0.38 for m in c['model']], marker='D', s=26, color=INK2,
                       edgecolor='white', linewidth=0.8, label='Cross-corpus (trained on the other corpus)', zorder=3)
        ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()))
        ax.set_title(corpus); ax.set_xlabel('Accuracy (mean, 95% CI)')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.grid(axis='y', visible=False)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / 'fig3_evaluation_ladder.png', bbox_inches='tight'); plt.close(fig)


def fig_confusions(conf, out, corpus='ISCX-URL-2016', model=ENS):
    plt = style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#0d366b'])
    c = conf[(conf['corpus'] == corpus) & (conf['experiment'] == 'headline') & (conf['model'] == model)]
    protos = [p for p in ['random', 'domain'] if p in set(c['protocol'])]
    fig, axes = plt.subplots(1, len(protos), figsize=(6.4, 3.0))
    for ax, proto in zip(np.atleast_1d(axes), protos):
        m = c[c['protocol'] == proto].groupby(['true', 'pred'])['count'].sum().unstack(fill_value=0)
        norm = m.div(m.sum(axis=1), axis=0)
        ax.imshow(norm.to_numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns)
        ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.grid(False)
        ax.set_title(f'{PROTOCOL_LABEL[proto]}')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = norm.iloc[i, j]
                ax.text(j, i, f'{v:.1%}\n({m.iloc[i, j]:,})', ha='center', va='center', fontsize=7.5,
                        color='white' if v > 0.55 else INK)
    fig.tight_layout(); fig.savefig(out / 'fig4_confusion_matrices.png', bbox_inches='tight'); plt.close(fig)


def fig_roc(roc, out, corpus='ISCX-URL-2016'):
    plt = style()
    r = roc[(roc['corpus'] == corpus) & (roc['experiment'] == 'headline')]
    protos = [p for p in ['random', 'domain'] if p in set(r['protocol'])]
    fig, axes = plt.subplots(1, len(protos), figsize=(6.4, 3.0), sharey=True)
    for ax, proto in zip(np.atleast_1d(axes), protos):
        for _, row in r[r['protocol'] == proto].iterrows():
            ax.plot(row['fpr'], row['tpr'], lw=2, color=CLASS_COLOR.get(row['class'], BLUE), label=f"{row['class']} vs rest")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--')
        ax.set_title(f'{PROTOCOL_LABEL[proto]} (first split)'); ax.set_xlabel('False positive rate')
        ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.legend(loc='lower right', fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel('True positive rate')
    fig.tight_layout(); fig.savefig(out / 'fig5_roc_curves.png', bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(HERE / 'results'))
    ap.add_argument('--no-audit', action='store_true', help='skip the corpus audit (needs both corpora on disk)')
    args = ap.parse_args()
    out = Path(args.dir)
    raw = out / 'raw'
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)

    import catboost, sklearn, sys, torch
    (raw / 'versions.json').write_text(json.dumps({
        'python': sys.version.split()[0], 'scikit-learn': sklearn.__version__, 'catboost': catboost.__version__,
        'torch': torch.__version__.split('+')[0], 'pandas': pd.__version__, 'numpy': np.__version__}, indent=2))

    sp = read_jsonl(raw / 'splits.jsonl')
    sp['roc_auc'] = pd.to_numeric(sp['roc_auc'], errors='coerce')
    within = sp[sp['protocol'] != 'cross-corpus']
    summ = summary_table(within)
    summ.to_csv(out / 'summary_by_protocol.csv', index=False)
    inflation_table(summ).to_csv(out / 'inflation.csv', index=False)
    paired_table(within, FEATURE_MODELS + [NGRAM_NAME, ARTEFACT_NAME]).to_csv(out / 'paired_tests.csv', index=False)
    per_class_table(within, [ENS, 'Random Forest', NGRAM_NAME]).to_csv(out / 'per_class.csv', index=False)
    side = [paired_against_headline(within, e, s) for e, s in
            [('tuned', ' (tuned)'), ('with_rank', ''), ('unescaped', '')]]
    side = [s for s in side if not s.empty]
    if side:
        pd.concat(side, ignore_index=True).to_csv(out / 'sensitivity_vs_headline.csv', index=False)
    cross = cross_table(sp)
    cross.to_csv(out / 'cross_corpus.csv', index=False)

    conf = read_jsonl(raw / 'confusions.jsonl')
    pooled = conf.groupby(['corpus', 'experiment', 'protocol', 'model', 'true', 'pred'], sort=False)['count'].sum().reset_index()
    pooled.to_csv(out / 'confusion_pooled.csv', index=False)

    sel = read_jsonl(raw / 'selected.jsonl')
    if not sel.empty:
        e = sel.explode('features')
        tot = sel.groupby(['corpus', 'experiment', 'protocol']).size().rename('splits')
        freq = e.groupby(['corpus', 'experiment', 'protocol', 'features']).size().rename('selected_in').reset_index()
        freq = freq.join(tot, on=['corpus', 'experiment', 'protocol'])
        freq['share'] = freq['selected_in'] / freq['splits']
        freq.sort_values(['corpus', 'experiment', 'protocol', 'share'], ascending=[True, True, True, False]) \
            .to_csv(out / 'feature_selection_frequency.csv', index=False)

    err, tot = read_jsonl(raw / 'errors.jsonl'), read_jsonl(raw / 'test_domains.jsonl')
    if not tot.empty:
        keys = ['corpus', 'protocol']
        hl_tot = tot[tot['experiment'] == 'headline']
        hl_err = err[err['experiment'] == 'headline'] if not err.empty else pd.DataFrame(columns=list(tot.columns) + ['errors', 'pred'])
        d_tot = hl_tot.groupby(keys + ['domain', 'suffix', 'true'])['n'].sum().rename('test_urls')
        d_err = hl_err.groupby(keys + ['domain', 'suffix', 'true'])['errors'].sum()
        dom = pd.concat([d_tot, d_err], axis=1).fillna({'errors': 0}).reset_index()
        dom['error_rate'] = dom['errors'] / dom['test_urls']
        dom.sort_values(keys + ['errors'], ascending=[True, True, False]).groupby(keys).head(25) \
            .to_csv(out / 'errors_top_domains.csv', index=False)
        suf = dom.groupby(keys + ['suffix', 'true'])[['test_urls', 'errors']].sum().reset_index()
        suf['error_rate'] = suf['errors'] / suf['test_urls']
        suf.sort_values(keys + ['test_urls'], ascending=[True, True, False]).to_csv(out / 'errors_by_suffix.csv', index=False)
        cls = dom.groupby(keys + ['true']).apply(lambda g: pd.Series({
            'test_urls': g['test_urls'].sum(), 'errors': g['errors'].sum(),
            'share_of_errors_from_top10_domains': g.nlargest(10, 'errors')['errors'].sum() / max(g['errors'].sum(), 1)}),
            include_groups=False).reset_index()
        cls.to_csv(out / 'errors_concentration.csv', index=False)

    fig_pipeline(figs)
    fig_ladder(summ, cross, figs)
    if not conf.empty:
        fig_confusions(conf, figs)
    roc = read_jsonl(raw / 'roc.jsonl')
    if not roc.empty:
        fig_roc(roc, figs)
    if not args.no_audit:
        audit, curves = corpus_audit()
        audit.to_csv(out / 'corpus_audit.csv', index=False)
        fig_concentration(curves, figs)
    print(f"tables and figures written to {out}")


if __name__ == '__main__':
    main()

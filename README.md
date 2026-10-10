# Soft-voting ensemble for spam and phishing URL detection

Code and result files for the paper *Soft-Voting Ensemble Learning for Spam and Phishing URL
Detection: A Host-Disjoint Evaluation*.

The pipeline extracts 50 features from each URL (whole URL, domain, directory and query
parameters), keeps the 20 most discriminative by Chi-squared selection, and trains Logistic
Regression, Decision Tree, Random Forest, CatBoost and a soft-voting ensemble of the four.
It evaluates them two ways:

- **Random split**: a stratified 80/20 split of URLs.
- **Host-disjoint splits**: five 80/20 splits in which no website in the test set appears in
  training.

## Results

Benign, phishing and spam URLs of ISCX-URL-2016 (57,343 URLs; 45,874 train / 11,469 test).

| Model | Random-split accuracy | Random-split ROC-AUC | Host-disjoint accuracy (mean ± SD) |
|---|---|---|---|
| Logistic Regression | 0.9350 | 0.9828 | 0.8351 ± 0.0321 |
| Decision Tree | 0.9932 | 0.9928 | 0.8148 ± 0.0584 |
| Random Forest | 0.9946 | 0.9997 | 0.8368 ± 0.0426 |
| CatBoost | 0.9936 | 0.9997 | 0.8693 ± 0.0557 |
| Soft-voting ensemble | 0.9944 | 0.9996 | 0.8625 ± 0.0175 |

Under the random split, 93.1% of test URLs come from a host that also occurs in training,
which is why the two columns differ so much. Full tables, including the runs without the
web-traffic-rank feature, are in [`results/`](results/).

## Reproducing

Tested with Python 3.14 and the package versions in `requirements.txt`.

```bash
pip install -r requirements.txt
```

1. **Data.** Download ISCX-URL-2016 from the Canadian Institute for Cybersecurity
   (<https://www.unb.ca/cic/datasets/url-2016.html>) and copy these three files from
   `FinalDataset/URL/` into `data/`:
   `Benign_list_big_final.csv`, `phishing_dataset.csv`, `spam_dataset.csv`.
2. **Rank list.** Download the Tranco list with ID `8PX5V`
   (<https://tranco-list.eu/list/8PX5V>) and save the zip in `resources/` under a name
   starting with `tranco`, for example `resources/tranco_top-1m_8PX5V.csv.zip`.
3. **Run.**

```bash
python url_ensemble_pipeline.py
```

The run takes roughly 15 minutes on a desktop CPU and writes to `results/`:

| File | Contents |
|---|---|
| `table1_metrics.csv` | Random split, all five models |
| `metrics_all_experiments.csv` | Random split with and without the rank feature |
| `host_disjoint_splits.csv` | Every model on every host-disjoint split |
| `host_disjoint_summary.csv` | Mean and standard deviation over the five splits |
| `chi2_feature_scores.csv` | Chi-squared score of each feature and whether it was selected |
| `confusion_*.csv`, `confusion_*.png` | Confusion matrices on the random split |
| `classification_report_ensemble.txt` | Per-class precision, recall and F1 of the ensemble |
| `run_summary.json` | Row counts, selected features, host statistics, package versions |

Results are seeded (`SEED = 42`); fit and prediction times vary between machines.

## Data sources

- Mamun, M. S. I., Rathore, M. A., Lashkari, A. H., Stakhanova, N., & Ghorbani, A. A. (2016).
  Detecting malicious URLs using lexical analysis. *Network and System Security*, 467–482.
  <https://doi.org/10.1007/978-3-319-46298-1_30>
- Le Pochat, V., Van Goethem, T., Tajalizadehkhoob, S., Korczyński, M., & Joosen, W. (2019).
  Tranco: A research-oriented top sites ranking hardened against manipulation. *NDSS 2019*.
  <https://doi.org/10.14722/ndss.2019.23386>

Put the URL corpus here, then run:

    python url_ensemble_pipeline.py

The reported results use the three raw URL lists of ISCX-URL-2016
(https://www.unb.ca/cic/datasets/url-2016.html), found in FinalDataset/URL/
of the archive:

    Benign_list_big_final.csv    35,378 URLs
    phishing_dataset.csv          9,965 URLs
    spam_dataset.csv             12,000 URLs

Accepted layouts (all *.csv files in this folder are read):

  A) One file per class, one URL per line, no header. The class is taken from the
     file name, which must contain "benign" (or "alexa" / "legit"), "phish" or "spam".

  B) A file with a header row containing a URL column (url / text / link) and a
     label column (label / type / class / target) whose values are
     benign / phishing / spam.

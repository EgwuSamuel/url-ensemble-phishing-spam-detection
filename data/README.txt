The two corpora are not redistributed here. Download them and place the files as follows.

data/iscx/
    ISCX-URL-2016, from the Canadian Institute for Cybersecurity
    (https://www.unb.ca/cic/datasets/url-2016.html). Copy these three files from
    FinalDataset/URL/ of the archive:

        Benign_list_big_final.csv    35,378 URLs
        phishing_dataset.csv          9,965 URLs
        spam_dataset.csv             12,000 URLs

    One URL per line, no header; the class is taken from the file name.

data/phiusiil/
    PhiUSIIL, from the UCI Machine Learning Repository
    (https://archive.ics.uci.edu/dataset/967):

        PhiUSIIL_Phishing_URL_Dataset.csv    235,795 rows

    Only the URL and label columns are read. In this file label 1 means legitimate
    and label 0 means phishing.

resources/  (optional, only for the web-traffic-rank sensitivity analysis)
    The Tranco list with ID 8PX5V (https://tranco-list.eu/list/8PX5V), saved as a zip
    whose name starts with "tranco".

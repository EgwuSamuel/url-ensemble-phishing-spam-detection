"""Character-level baselines that read the raw URL string instead of engineered features.

  * character n-gram TF-IDF + logistic regression (scikit-learn)
  * a small character-level CNN (PyTorch, CPU)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

NGRAM_NAME = 'Char n-gram LR'
CNN_NAME = 'Char-CNN'


def make_ngram_model(seed=42):
    """TF-IDF over character 1- to 3-grams, then logistic regression fitted by SGD."""
    return make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(1, 3), min_df=3, max_features=200_000,
                        sublinear_tf=True, lowercase=False, dtype=np.float32),
        SGDClassifier(loss='log_loss', alpha=1e-6, max_iter=20, tol=None, random_state=seed, n_jobs=-1))


# ----------------------------------------------------------------------------- CNN
PAD, UNK, VOCAB = 0, 96, 97     # printable ASCII 32..126 -> 1..95


def encode_urls(urls, max_len):
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, u in enumerate(urls):
        codes = np.frombuffer(str(u)[:max_len].encode('utf-8', 'replace')[:max_len], dtype=np.uint8).astype(np.int64)
        idx = np.where((codes >= 32) & (codes <= 126), codes - 31, UNK)
        out[i, :len(idx)] = idx
    return out


def fit_predict_cnn(train_urls, y_train, test_urls, n_classes, epochs=2, max_len=128,
                    batch_size=256, seed=42, threads=4):
    """Train the CNN on train_urls and return class probabilities for test_urls."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(threads)

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 32, padding_idx=PAD)
            self.convs = nn.ModuleList([nn.Conv1d(32, 96, k, padding=k // 2) for k in (3, 4, 5)])
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(96 * 3, n_classes)

        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
            return self.out(self.drop(h))

    Xtr = torch.from_numpy(encode_urls(list(train_urls), max_len))
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    Xte = torch.from_numpy(encode_urls(list(test_urls), max_len))
    model = CharCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xte), 2048):
            probs.append(torch.softmax(model(Xte[s:s + 2048]), dim=1).numpy())
    return np.vstack(probs)

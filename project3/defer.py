import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from project3.expert import simulate_expert_batch


def compute_defer_targets(ai_preds, expert_preds, true_labels):
    ai_preds = np.asarray(ai_preds)
    expert_preds = np.asarray(expert_preds)
    true_labels = np.asarray(true_labels)
    return ((expert_preds == true_labels) & (ai_preds != true_labels)).astype(int)


def train_defer_model(train_texts, defer_targets):
    # Fit a fresh TF-IDF vectorizer (rather than reusing the baseline's) so the deferral
    # model is a fully independent pipeline -- simpler to reason about / swap out, and the
    # cost of a second fit_transform over 120k docs is negligible next to training itself.
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(train_texts)
    # defer_targets is heavily imbalanced (only a few % of rows are "deferring would
    # help"); without class_weight='balanced' the model just always predicts the
    # majority class and P(defer=1) never crosses 0.5.
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, defer_targets)
    return vectorizer, model


def defer_predict_proba(defer_vectorizer, defer_model, texts):
    X = defer_vectorizer.transform(texts)
    # defer_targets may be all-0/all-1 in small active-learning subsets, in which case
    # sklearn only learns one class -- guard so callers always get a well-formed P(defer=1).
    if len(defer_model.classes_) == 1:
        only_class = defer_model.classes_[0]
        fill = 1.0 if only_class == 1 else 0.0
        return np.full(X.shape[0], fill)
    class_idx = list(defer_model.classes_).index(1)
    return defer_model.predict_proba(X)[:, class_idx]


def combine_predictions(ai_preds, expert_preds, defer_prob, threshold=0.5):
    ai_preds = np.asarray(ai_preds)
    expert_preds = np.asarray(expert_preds)
    defer_mask = defer_prob > threshold
    final_preds = np.where(defer_mask, expert_preds, ai_preds)
    return final_preds, defer_mask


def system_accuracy_at_threshold(ai_preds, expert_preds, true_labels, defer_prob, threshold):
    final_preds, defer_mask = combine_predictions(ai_preds, expert_preds, defer_prob, threshold)
    acc = accuracy_score(true_labels, final_preds)
    deferral_rate = float(np.mean(defer_mask))
    return acc, deferral_rate


def oracle_accuracy(ai_preds, expert_preds, true_labels):
    ai_preds = np.asarray(ai_preds)
    expert_preds = np.asarray(expert_preds)
    true_labels = np.asarray(true_labels)
    would_help = (expert_preds == true_labels) & (ai_preds != true_labels)
    final_preds = np.where(would_help, expert_preds, ai_preds)
    return accuracy_score(true_labels, final_preds)


def coverage_accuracy_curve(ai_preds, expert_preds, true_labels, defer_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 15)
    curve = []
    for t in thresholds:
        acc, rate = system_accuracy_at_threshold(ai_preds, expert_preds, true_labels, defer_prob, t)
        curve.append((float(t), float(acc), float(rate)))
    return curve


def best_threshold(curve):
    # The natural 0.5 cutoff is miscalibrated when defer_target is rare (~3% positive),
    # so the reported "system" operating point is the threshold maximizing accuracy on
    # the sweep already computed by coverage_accuracy_curve, not a hardcoded 0.5.
    return max(curve, key=lambda row: row[1])[0]

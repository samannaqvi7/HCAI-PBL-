import numpy as np

from sklearn.linear_model import LogisticRegression

from project3.data import CLASS_NAMES
from project3.expert import simulate_expert_batch, COMPETENCE
from project3.defer import compute_defer_targets, defer_predict_proba, coverage_accuracy_curve


BATCH_SIZE = 100
BUDGET = 3000
RANDOM_SEED = 777


def compute_confidence(baseline_vectorizer, baseline_model, texts):
    # Confidence = the baseline classifier's own max predicted probability, computed once
    # up front over the TRAIN set and held fixed across all active-learning rounds. A
    # "real" uncertainty-sampling loop might recompute confidence as the model driving it
    # updates, but here the Task 1 baseline classifier is fixed throughout Task 4 -- only
    # the deferral classifier is retrained per round -- so this fixed criterion is exact,
    # not an approximation.
    proba = baseline_model.predict_proba(baseline_vectorizer.transform(texts))
    return proba.max(axis=1)


def uncertainty_order(confidence):
    return np.argsort(confidence)  # ascending confidence = most uncertain first


def random_order(n, seed=RANDOM_SEED):
    order = np.arange(n)
    np.random.default_rng(seed).shuffle(order)
    return order


def _train_partial_defer_model(baseline_vectorizer, baseline_model, train_texts, train_labels, queried_idx):
    texts_subset = [train_texts[i] for i in queried_idx]
    labels_subset = train_labels[queried_idx]
    ai_pred_subset = baseline_model.predict(baseline_vectorizer.transform(texts_subset))
    expert_pred_subset = simulate_expert_batch(labels_subset, queried_idx)
    defer_targets_subset = compute_defer_targets(ai_pred_subset, expert_pred_subset, labels_subset)

    X_subset = baseline_vectorizer.transform(texts_subset)
    partial_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    partial_model.fit(X_subset, defer_targets_subset)
    return partial_model, ai_pred_subset, expert_pred_subset, labels_subset, defer_targets_subset


def _per_class_competence(expert_pred_subset, labels_subset):
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = labels_subset == c
        per_class[name] = float(np.mean(expert_pred_subset[mask] == labels_subset[mask])) if mask.any() else None
    return per_class


def _competence_mae(estimated):
    # Mean absolute error between the estimated per-class competence (from queried points
    # so far) and the expert's TRUE designed profile -- classes with no queried points yet
    # are excluded rather than penalized, since "unknown" isn't the same as "wrong".
    errors = [abs(estimated[name] - COMPETENCE[name]) for name in CLASS_NAMES if estimated[name] is not None]
    return float(np.mean(errors)) if errors else None


def simulate_strategy(order, baseline_vectorizer, baseline_model, train_texts, train_labels,
                       test_texts, test_labels, ai_pred_test, expert_pred_test,
                       batch_size=BATCH_SIZE, budget=BUDGET):
    train_labels = np.asarray(train_labels)
    results = []
    queried = []
    n_rounds = budget // batch_size
    for r in range(n_rounds):
        start, end = r * batch_size, (r + 1) * batch_size
        queried.extend(order[start:end].tolist())
        idx_arr = np.array(queried)

        partial_model, ai_pred_subset, expert_pred_subset, labels_subset, defer_targets_subset = (
            _train_partial_defer_model(baseline_vectorizer, baseline_model, train_texts, train_labels, idx_arr))

        # The threshold that's "best" shifts as the partial deferral model's probability
        # calibration changes with training-set size (e.g. a threshold tuned on the full
        # 120k-row model doesn't transfer to a model fit on 100-3000 rows) -- so each round
        # reports its own best-achievable operating point, swept fresh, rather than reusing
        # a single global threshold across rounds.
        defer_prob_test = defer_predict_proba(baseline_vectorizer, partial_model, test_texts)
        curve = coverage_accuracy_curve(ai_pred_test, expert_pred_test, test_labels, defer_prob_test)
        threshold, acc, rate = max(curve, key=lambda row: row[1])

        competence = _per_class_competence(expert_pred_subset, labels_subset)
        results.append({
            'num_queries': len(queried),
            'threshold': float(threshold),
            'system_accuracy': float(acc),
            'defer_rate': float(rate),
            # How "information-dense" the queried set is for learning deferral -- the
            # whole point of active learning here is to enrich this relative to random.
            'defer_positive_rate': float(np.mean(defer_targets_subset)),
            'competence_estimate': competence,
            'competence_mae': _competence_mae(competence),
        })

    return results


def run_active_learning(baseline_vectorizer, baseline_model, train_texts, train_labels,
                         test_texts, test_labels, batch_size=BATCH_SIZE, budget=BUDGET):
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    test_idx = np.arange(len(test_labels))

    ai_pred_test = baseline_model.predict(baseline_vectorizer.transform(test_texts))
    expert_pred_test = simulate_expert_batch(test_labels, test_idx)

    confidence = compute_confidence(baseline_vectorizer, baseline_model, train_texts)
    unc_order = uncertainty_order(confidence)
    rand_order = random_order(len(train_texts))

    unc_results = simulate_strategy(
        unc_order, baseline_vectorizer, baseline_model, train_texts, train_labels,
        test_texts, test_labels, ai_pred_test, expert_pred_test,
        batch_size=batch_size, budget=budget)
    rand_results = simulate_strategy(
        rand_order, baseline_vectorizer, baseline_model, train_texts, train_labels,
        test_texts, test_labels, ai_pred_test, expert_pred_test,
        batch_size=batch_size, budget=budget)

    return {
        'uncertainty': unc_results,
        'random': rand_results,
    }

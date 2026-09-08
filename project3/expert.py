import numpy as np

from project3.data import CLASS_NAMES


COMPETENCE = {'Sports': 0.95, 'Sci/Tech': 0.95, 'World': 0.40, 'Business': 0.40}
BASE_SEED = 12345

# When the expert errs, the wrong label is drawn from the other 3 classes with these
# per-true-class weights (index-aligned with CLASS_NAMES). World and Business are each
# other's most likely confusion (topically overlapping in AG News -- e.g. trade/economic
# stories can read as either), so each one's error mass is skewed toward the other.
_ERROR_WEIGHTS = {
    'World':    {'Sports': 0.15, 'Business': 0.70, 'Sci/Tech': 0.15},
    'Sports':   {'World': 0.34, 'Business': 0.33, 'Sci/Tech': 0.33},
    'Business': {'World': 0.70, 'Sports': 0.15, 'Sci/Tech': 0.15},
    'Sci/Tech': {'World': 0.34, 'Sports': 0.33, 'Business': 0.33},
}


def simulate_expert_one(true_label, index):
    rng = np.random.default_rng(BASE_SEED + index)
    true_name = CLASS_NAMES[true_label]
    if rng.random() < COMPETENCE[true_name]:
        return true_label
    weights = _ERROR_WEIGHTS[true_name]
    other_names = list(weights.keys())
    probs = np.array([weights[n] for n in other_names])
    probs = probs / probs.sum()
    chosen_name = rng.choice(other_names, p=probs)
    return CLASS_NAMES.index(chosen_name)


def simulate_expert_batch(true_labels, indices):
    return np.array([simulate_expert_one(t, i) for t, i in zip(true_labels, indices)])


def evaluate_expert(true_labels, indices):
    true_labels = np.asarray(true_labels)
    indices = np.asarray(indices)
    preds = simulate_expert_batch(true_labels, indices)
    overall_acc = float(np.mean(preds == true_labels))
    per_class_acc = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = true_labels == c
        per_class_acc[name] = float(np.mean(preds[mask] == true_labels[mask])) if mask.any() else None
    return overall_acc, per_class_acc

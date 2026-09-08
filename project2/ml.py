import numpy as np
import pandas as pd
from palmerpenguins import load_penguins

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


NUMERIC_FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
ISLANDS = ['Biscoe', 'Dream', 'Torgersen']
SEXES = ['female', 'male']
YEARS = [2007, 2008, 2009]
CATEGORICAL_FEATURES = {'island': ISLANDS, 'sex': SEXES, 'year': YEARS}
FEATURE_COLUMNS = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())
TARGET = 'species'
SPECIES = ['Adelie', 'Chinstrap', 'Gentoo']
SPECIES_COLORS = {'Adelie': '#1f77b4', 'Chinstrap': '#ff7f0e', 'Gentoo': '#2ca02c'}
FEATURE_LABELS = {
    'bill_length_mm': 'Bill length (mm)',
    'bill_depth_mm': 'Bill depth (mm)',
    'flipper_length_mm': 'Flipper length (mm)',
    'body_mass_g': 'Body mass (g)',
}

# One-hot encode with C-1 dummy columns per categorical feature (each feature's first
# listed category is the implicit all-zeros default), per the lecture's convention.
ENCODED_COLUMNS = NUMERIC_FEATURES + [
    f"{feat}_{cat}" for feat, cats in CATEGORICAL_FEATURES.items() for cat in cats[1:]
]

LEAF_GRID = list(range(2, 31))
C_GRID = np.logspace(-2.5, 1.5, 16)
COEF_EPS = 1e-4


def load_clean_data():
    df = load_penguins()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)
    return df


def encode(df):
    parts = [df[NUMERIC_FEATURES].reset_index(drop=True).astype(float)]
    for feat, cats in CATEGORICAL_FEATURES.items():
        col = df[feat].reset_index(drop=True)
        for cat in cats[1:]:
            parts.append((col == cat).astype(float).rename(f"{feat}_{cat}"))
    X = pd.concat(parts, axis=1)
    return X[ENCODED_COLUMNS]


class Dataset:
    def __init__(self, random_state=42, test_size=0.25):
        self.df = load_clean_data()
        self.X = encode(self.df)
        self.y = self.df[TARGET].values
        idx = np.arange(len(self.df))
        self.idx_train, self.idx_test = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=self.y
        )
        self.X_train = self.X.iloc[self.idx_train].reset_index(drop=True)
        self.y_train = self.y[self.idx_train]
        self.X_test = self.X.iloc[self.idx_test].reset_index(drop=True)
        self.y_test = self.y[self.idx_test]
        self.df_train = self.df.iloc[self.idx_train].reset_index(drop=True)
        self.mad = {
            f: max(float(np.median(np.abs(self.df_train[f] - np.median(self.df_train[f])))), 1e-6)
            for f in NUMERIC_FEATURES
        }
        self.std = {f: max(float(self.df_train[f].std()), 1e-6) for f in NUMERIC_FEATURES}


def train_tree_family(ds):
    family = []
    for k in LEAF_GRID:
        model = DecisionTreeClassifier(max_leaf_nodes=k, random_state=42)
        model.fit(ds.X_train, ds.y_train)
        acc = accuracy_score(ds.y_test, model.predict(ds.X_test))
        family.append({'label': k, 'model': model, 'acc': acc, 'omega': model.get_n_leaves()})
    return family


def train_logreg_family(ds):
    family = []
    for c in C_GRID:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', solver='saga', C=c, max_iter=5000)),
        ])
        pipe.fit(ds.X_train, ds.y_train)
        acc = accuracy_score(ds.y_test, pipe.predict(ds.X_test))
        coef = pipe.named_steps['clf'].coef_
        omega = int(np.sum(np.abs(coef) > COEF_EPS))
        family.append({'label': round(float(c), 4), 'model': pipe, 'acc': acc, 'omega': omega})
    return family


# NOTE: these are process-local, module-level caches -- correct and sufficient for
# `manage.py runserver` (a single process), but they would NOT be shared across
# multiple worker processes in a real multi-process deployment (e.g. gunicorn with
# several workers), since each worker gets its own copy of this module's globals.
# That's fine for this project; just don't mistake this for a production-grade cache.
_dataset_cache = {}
_family_cache = {}


def get_dataset():
    if 'ds' not in _dataset_cache:
        _dataset_cache['ds'] = Dataset()
    return _dataset_cache['ds']


def get_family(model_type, ds):
    if model_type not in _family_cache:
        if model_type == 'tree':
            _family_cache[model_type] = train_tree_family(ds)
        else:
            _family_cache[model_type] = train_logreg_family(ds)
    return _family_cache[model_type]


def select_by_lambda(family, lam):
    return max(family, key=lambda r: r['acc'] - lam * r['omega'])


def sample_perturbations(x_row, n, numeric_std, cat_flip_prob, rng):
    data = {col: np.full(n, x_row[col], dtype=object) for col in FEATURE_COLUMNS}
    samples = pd.DataFrame(data)
    for f in NUMERIC_FEATURES:
        samples[f] = float(x_row[f]) + rng.normal(0, numeric_std[f], size=n)
    for feat, cats in CATEGORICAL_FEATURES.items():
        flip_mask = rng.random(n) < cat_flip_prob
        if flip_mask.any():
            choices = rng.choice(cats, size=int(flip_mask.sum()))
            samples.loc[flip_mask, feat] = choices
    return samples


def counterfactual_distance(samples_df, x_row, mad):
    dist = np.zeros(len(samples_df))
    for f in NUMERIC_FEATURES:
        dist += np.abs(samples_df[f].values.astype(float) - float(x_row[f])) / mad[f]
    for feat in CATEGORICAL_FEATURES:
        dist += (samples_df[feat].values != x_row[feat]).astype(float)
    return dist


def generate_counterfactuals(model, ds, x_row, target_label, k=5, max_rounds=6):
    rng = np.random.default_rng(42)
    n = 500
    noise_scale = 1.0
    cat_flip_prob = 0.3
    for _ in range(max_rounds):
        numeric_std = {f: ds.std[f] * noise_scale for f in NUMERIC_FEATURES}
        samples = sample_perturbations(x_row, n, numeric_std, cat_flip_prob, rng)
        preds = model.predict(encode(samples))
        matches = samples[preds == target_label].copy()
        if len(matches) > 0:
            matches['distance'] = counterfactual_distance(matches, x_row, ds.mad)
            matches = matches.sort_values('distance').head(k).reset_index(drop=True)
            return matches
        n *= 2
        noise_scale *= 1.5
        cat_flip_prob = min(cat_flip_prob * 1.3, 0.9)
    return None


def pdp_grid(ds, feature, n_points=30):
    lo, hi = ds.df[feature].min(), ds.df[feature].max()
    return np.linspace(lo, hi, n_points)


def compute_pdp(model, X_reference, feature, grid_values):
    classes = list(model.classes_)
    curves = {cls: np.zeros(len(grid_values)) for cls in classes}
    X_mod = X_reference.copy()
    for i, g in enumerate(grid_values):
        X_mod[feature] = g
        proba = model.predict_proba(X_mod)
        for c, cls in enumerate(classes):
            curves[cls][i] = proba[:, c].mean()
    return curves


def _bin_edges(values, n_bins):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)


def _center_cumulative(cum, counts):
    weights = counts / counts.sum()
    mean_cum = (cum * weights[:, None]).sum(axis=0)
    return cum - mean_cum


def compute_ale_tree(model, X_reference, feature, n_bins=12):
    values = X_reference[feature].values
    edges = _bin_edges(values, n_bins)
    n_bins_actual = len(edges) - 1
    bins = np.clip(np.digitize(values, edges[1:-1], right=True), 0, n_bins_actual - 1)
    classes = list(model.classes_)
    local_effects = np.zeros((n_bins_actual, len(classes)))
    counts = np.zeros(n_bins_actual)
    for b in range(n_bins_actual):
        mask = bins == b
        counts[b] = mask.sum()
        if counts[b] == 0:
            continue
        X_lo = X_reference[mask].copy()
        X_hi = X_reference[mask].copy()
        X_lo[feature] = edges[b]
        X_hi[feature] = edges[b + 1]
        p_lo = model.predict_proba(X_lo)
        p_hi = model.predict_proba(X_hi)
        local_effects[b] = (p_hi - p_lo).mean(axis=0)
    cum = np.cumsum(local_effects, axis=0)
    centered = _center_cumulative(cum, counts)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    return bin_centers, {cls: centered[:, i] for i, cls in enumerate(classes)}


def compute_ale_logreg(pipeline, X_reference, feature, n_bins=12):
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']
    feat_idx = list(X_reference.columns).index(feature)
    scale = scaler.scale_[feat_idx]

    values = X_reference[feature].values
    edges = _bin_edges(values, n_bins)
    n_bins_actual = len(edges) - 1
    bins = np.clip(np.digitize(values, edges[1:-1], right=True), 0, n_bins_actual - 1)
    classes = list(clf.classes_)
    W = clf.coef_
    local_effects = np.zeros((n_bins_actual, len(classes)))
    counts = np.zeros(n_bins_actual)
    for b in range(n_bins_actual):
        mask = bins == b
        counts[b] = mask.sum()
        if counts[b] == 0:
            continue
        Xb_scaled = scaler.transform(X_reference[mask])
        proba = clf.predict_proba(Xb_scaled)
        weighted_sum = proba @ W[:, feat_idx]
        deriv_z = proba * (W[:, feat_idx][None, :] - weighted_sum[:, None])
        deriv_x = deriv_z / scale
        bin_width = edges[b + 1] - edges[b]
        local_effects[b] = deriv_x.mean(axis=0) * bin_width
    cum = np.cumsum(local_effects, axis=0)
    centered = _center_cumulative(cum, counts)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    return bin_centers, {cls: centered[:, i] for i, cls in enumerate(classes)}


def compute_ale(model_type, model, X_reference, feature, n_bins=12):
    if model_type == 'logreg':
        return compute_ale_logreg(model, X_reference, feature, n_bins)
    return compute_ale_tree(model, X_reference, feature, n_bins)

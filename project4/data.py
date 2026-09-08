import numpy as np
import pandas as pd


CSV_URL = "https://raw.githubusercontent.com/sundeepblue/movie_rating_prediction/master/movie_metadata.csv"

NUMERIC_FEATURES = [
    'duration', 'title_year', 'imdb_score', 'num_voted_users', 'budget', 'gross',
    'director_facebook_likes', 'actor_1_facebook_likes', 'cast_total_facebook_likes',
    'movie_facebook_likes',
]
# These are heavily right-skewed (a handful of blockbusters dwarf everything else), so
# they're log1p-transformed before standardizing -- otherwise a few outliers would
# dominate the feature scale.
LOG_FEATURES = {'num_voted_users', 'budget', 'gross', 'director_facebook_likes',
                 'actor_1_facebook_likes', 'cast_total_facebook_likes', 'movie_facebook_likes'}

TOP_N_GENRES = 15
CONTENT_RATINGS = ['R', 'PG-13', 'PG', 'Not Rated', 'G', 'Unrated']  # + "Other" bucket

DISPLAY_COLUMNS = ['movie_title', 'genres', 'director_name', 'title_year', 'duration',
                   'imdb_score', 'content_rating']

_cache = {}


def _load_raw():
    if 'raw' not in _cache:
        _cache['raw'] = pd.read_csv(CSV_URL)
    return _cache['raw']


def _top_genres(df, n=TOP_N_GENRES):
    counts = {}
    for cell in df['genres'].dropna():
        for g in cell.split('|'):
            counts[g] = counts.get(g, 0) + 1
    return [g for g, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def build_dataset():
    """Returns (movies, X, feature_names): `movies` is a cleaned display DataFrame (one
    row per usable movie, index reset), `X` is the standardized feature matrix
    (numpy array, same row order as `movies`), `feature_names` names X's columns."""
    if 'dataset' in _cache:
        return _cache['dataset']

    df = _load_raw()
    df = df.dropna(subset=['genres', 'title_year']).reset_index(drop=True)
    df['movie_title'] = df['movie_title'].str.strip()
    # The raw CSV contains exact duplicate rows for some movies (e.g. "Eddie the Eagle"
    # appears twice, identical down to director/duration/score) -- dedup here, before
    # top_genres/X are built, so `movies` and `X` stay row-aligned everywhere downstream
    # (sample_indices, _movie_context, recommendation ranking) rather than risking a
    # later filter desyncing the two.
    df = df.drop_duplicates(subset=['movie_title']).reset_index(drop=True)

    top_genres = _top_genres(df)

    numeric = df[NUMERIC_FEATURES].copy()
    for col in LOG_FEATURES:
        numeric[col] = np.log1p(numeric[col].clip(lower=0))
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    genre_cols = {}
    for g in top_genres:
        genre_cols[f'genre_{g}'] = df['genres'].fillna('').apply(lambda s, g=g: float(g in s.split('|')))
    genre_df = pd.DataFrame(genre_cols)

    rating = df['content_rating'].where(df['content_rating'].isin(CONTENT_RATINGS), 'Other')
    rating_cols = {f'rating_{r}': (rating == r).astype(float) for r in CONTENT_RATINGS + ['Other']}
    rating_df = pd.DataFrame(rating_cols)

    feature_df = pd.concat([numeric.reset_index(drop=True), genre_df, rating_df], axis=1)
    feature_names = list(feature_df.columns)

    # Standardize the whole matrix (including the 0/1 genre/rating columns) so every
    # dimension has comparable scale under the shared L2 prior used when fitting w.
    mean = feature_df.mean()
    std = feature_df.std().replace(0, 1)
    X = ((feature_df - mean) / std).to_numpy()

    movies = df[DISPLAY_COLUMNS].reset_index(drop=True)

    _cache['dataset'] = (movies, X, feature_names)
    return _cache['dataset']


def sample_indices(n, rng, exclude=()):
    movies, X, _ = build_dataset()
    pool = np.setdiff1d(np.arange(len(movies)), np.asarray(list(exclude), dtype=int))
    return rng.choice(pool, size=n, replace=False)

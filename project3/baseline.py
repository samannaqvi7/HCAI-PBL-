from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_baseline(train_texts, train_labels):
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    # lbfgs needs more than the default 100 iters to converge on 20k sparse TF-IDF features.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)
    return vectorizer, model


def evaluate_baseline(vectorizer, model, test_texts, test_labels):
    X_test = vectorizer.transform(test_texts)
    preds = model.predict(X_test)
    return accuracy_score(test_labels, preds)

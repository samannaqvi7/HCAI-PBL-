import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


UPLOAD_PATH = os.path.join(settings.MEDIA_ROOT, 'uploaded_data.csv')
FILENAME_PATH = os.path.join(settings.MEDIA_ROOT, 'uploaded_filename.txt')
PLOT_PATH = os.path.join(settings.MEDIA_ROOT, 'plot.png')
SWEEP_PATH = os.path.join(settings.MEDIA_ROOT, 'sweep.png')

MODELS = {
    'dtree': {
        'label': 'Decision Tree',
        'hyperparam_label': 'Max depth',
        'hyperparam_default': 3,
        'hyperparam_min': 1,
        'hyperparam_max': 20,
    },
    'logreg': {
        'label': 'Logistic Regression',
        'hyperparam_label': 'Regularization strength (C)',
        'hyperparam_default': 1.0,
        'hyperparam_min': 0.01,
        'hyperparam_max': 10,
    },
    'knn': {
        'label': 'k-Nearest Neighbors',
        'hyperparam_label': 'Number of neighbors (k)',
        'hyperparam_default': 5,
        'hyperparam_min': 1,
        'hyperparam_max': 50,
    },
}

MODEL_BUILDERS = {
    'dtree': lambda hp: DecisionTreeClassifier(max_depth=int(float(hp)), random_state=42),
    'logreg': lambda hp: LogisticRegression(C=float(hp), max_iter=1000),
    'knn': lambda hp: KNeighborsClassifier(n_neighbors=int(float(hp))),
}

SCORES = {
    'accuracy': 'Accuracy',
    'precision': 'Precision (macro)',
    'recall': 'Recall (macro)',
    'f1': 'F1 score (macro)',
}

SCORE_FUNCS = {
    'accuracy': lambda yt, yp: accuracy_score(yt, yp),
    'precision': lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0),
    'recall': lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0),
    'f1': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
}

TEST_SIZE_OPTIONS = [10, 20, 30, 40, 50]


def _make_scatter_plot(df, label_column, x_feature, y_feature):
    fig, ax = plt.subplots()
    label_codes, label_names = pd.factorize(df[label_column])
    scatter = ax.scatter(df[x_feature], df[y_feature], c=label_codes, cmap='viridis')
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title('Data visualization')
    ax.legend(scatter.legend_elements()[0], label_names, title=label_column)
    fig.savefig(PLOT_PATH)
    plt.close(fig)


def _dataset_summary(df, filename, feature_columns, label_column):
    stats = df[feature_columns].describe().round(2)
    feature_stats = [
        {
            'feature': f,
            'mean': stats.loc['mean', f],
            'std': stats.loc['std', f],
            'min': stats.loc['min', f],
            'max': stats.loc['max', f],
        }
        for f in feature_columns
    ]
    preview_columns = feature_columns + [label_column]
    return {
        'filename': filename,
        'n_rows': len(df),
        'n_features': len(feature_columns),
        'feature_stats': feature_stats,
        'label_column': label_column,
        'class_counts': df[label_column].value_counts().to_dict(),
        'preview_columns': preview_columns,
        'preview_rows': df[preview_columns].head(5).values.tolist(),
    }


def _hyperparam_sweep_values(model_key):
    lo = MODELS[model_key]['hyperparam_min']
    hi = MODELS[model_key]['hyperparam_max']
    if model_key == 'logreg':
        # C is conventionally varied on a log scale
        exponents = np.linspace(np.log10(lo), np.log10(hi), 8)
        return [round(float(10 ** e), 3) for e in exponents]
    values = {int(round(v)) for v in np.linspace(lo, hi, 8)}
    return sorted(values)


def _make_sweep_plot(values, scores, hyperparam_label, score_label, best_index):
    fig, ax = plt.subplots()
    ax.plot(values, scores, marker='o')
    ax.scatter([values[best_index]], [scores[best_index]], color='red', zorder=3, label='Best')
    ax.set_xlabel(hyperparam_label)
    ax.set_ylabel(score_label)
    ax.set_title('Score vs hyperparameter')
    ax.legend()
    fig.savefig(SWEEP_PATH)
    plt.close(fig)


def _feature_importance(model, model_key, feature_columns):
    if model_key == 'dtree':
        rows = [
            {'feature': f, 'value': round(v, 3)}
            for f, v in zip(feature_columns, model.feature_importances_)
        ]
        rows.sort(key=lambda r: r['value'], reverse=True)
        return {'type': 'single', 'rows': rows}

    if model_key == 'logreg':
        classes = list(model.classes_)
        coefs = model.coef_
        if coefs.shape[0] == 1:
            # binary classification: sklearn only stores coefficients for the positive class
            classes = [classes[1]]
        rows = [
            {'feature': f, 'values': [round(coefs[c][i], 3) for c in range(len(classes))]}
            for i, f in enumerate(feature_columns)
        ]
        return {'type': 'per_class', 'classes': classes, 'rows': rows}

    return {'type': 'unsupported'}


def index(request):
    context = {
        'models': MODELS,
        'scores': SCORES,
        'test_size_options': TEST_SIZE_OPTIONS,
        'selected_model': 'dtree',
        'selected_hyperparam': MODELS['dtree']['hyperparam_default'],
        'selected_test_size': 20,
        'selected_score': 'accuracy',
        'hyperparam_label': MODELS['dtree']['hyperparam_label'],
    }

    if request.method != 'POST':
        return render(request, 'project1/index.html', context)

    action = request.POST.get('action')

    if action == 'upload':
        if not request.FILES.get('csv_file'):
            context['upload_error'] = 'Please choose a CSV file before clicking Upload.'
            return render(request, 'project1/index.html', context)
        uploaded_file = request.FILES['csv_file']
        df = pd.read_csv(uploaded_file)
        df.to_csv(UPLOAD_PATH, index=False)
        with open(FILENAME_PATH, 'w') as f:
            f.write(uploaded_file.name)
        filename = uploaded_file.name
    elif action in ('plot', 'train', 'sweep'):
        if not os.path.exists(UPLOAD_PATH):
            context['upload_error'] = 'Please upload a dataset first.'
            return render(request, 'project1/index.html', context)
        df = pd.read_csv(UPLOAD_PATH)
        filename = open(FILENAME_PATH).read().strip() if os.path.exists(FILENAME_PATH) else 'uploaded_data.csv'
    else:
        return render(request, 'project1/index.html', context)

    feature_columns = list(df.columns[:-1])
    label_column = df.columns[-1]

    x_feature = request.POST.get('x_feature') or feature_columns[0]
    y_feature = request.POST.get('y_feature') or feature_columns[1]
    _make_scatter_plot(df, label_column, x_feature, y_feature)

    context.update({
        'plot_url': f"{settings.MEDIA_URL}plot.png?v={int(os.path.getmtime(PLOT_PATH))}",
        'feature_columns': feature_columns,
        'x_feature': x_feature,
        'y_feature': y_feature,
        'dataset_info': _dataset_summary(df, filename, feature_columns, label_column),
        'upload_open': action == 'upload',
    })

    if request.POST.get('action') == 'train':
        model_key = request.POST.get('model', 'dtree')
        hyperparam = request.POST.get('hyperparam', MODELS[model_key]['hyperparam_default'])
        test_size = int(request.POST.get('test_size', 20))
        score_key = request.POST.get('score', 'accuracy')

        context.update({
            'selected_model': model_key,
            'selected_hyperparam': hyperparam,
            'selected_test_size': test_size,
            'selected_score': score_key,
            'hyperparam_label': MODELS[model_key]['hyperparam_label'],
        })

        try:
            X = df[feature_columns]
            y = df[label_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=y
            )

            model = MODEL_BUILDERS[model_key](hyperparam)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            context['result'] = round(SCORE_FUNCS[score_key](y_test, y_pred), 3)
            context['score_label'] = SCORES[score_key]
            context['has_result'] = True
            context['feature_importance'] = _feature_importance(model, model_key, feature_columns)
        except ValueError as e:
            context['error'] = str(e)

    elif request.POST.get('action') == 'sweep':
        model_key = request.POST.get('model', 'dtree')
        test_size = int(request.POST.get('test_size', 20))
        score_key = request.POST.get('score', 'accuracy')

        context.update({
            'selected_model': model_key,
            'selected_test_size': test_size,
            'selected_score': score_key,
            'hyperparam_label': MODELS[model_key]['hyperparam_label'],
        })

        try:
            X = df[feature_columns]
            y = df[label_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=y
            )

            values = _hyperparam_sweep_values(model_key)
            scores = []
            for v in values:
                model = MODEL_BUILDERS[model_key](v)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(round(SCORE_FUNCS[score_key](y_test, y_pred), 3))

            best_index = max(range(len(scores)), key=lambda i: scores[i])
            _make_sweep_plot(values, scores, MODELS[model_key]['hyperparam_label'], SCORES[score_key], best_index)

            context.update({
                'score_label': SCORES[score_key],
                'sweep_results': list(zip(values, scores)),
                'sweep_best_value': values[best_index],
                'sweep_best_score': scores[best_index],
                'sweep_plot_url': f"{settings.MEDIA_URL}sweep.png?v={int(os.path.getmtime(SWEEP_PATH))}",
                'selected_hyperparam': values[best_index],
            })
        except ValueError as e:
            context['error'] = str(e)

    return render(request, 'project1/index.html', context)

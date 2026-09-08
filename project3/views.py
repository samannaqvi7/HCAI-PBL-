import json
import os

import joblib
import numpy as np

from django.http import FileResponse, Http404
from django.shortcuts import render

from project3 import precompute
from project3.data import CLASS_NAMES
from project3.defer import defer_predict_proba
from project3.expert import simulate_expert_batch, COMPETENCE
from project3.report import REPORT_PATH


# Process-local cache: these artifacts are built once by `python manage.py build_project3`
# (AG News' 120k-row scale makes retraining per-request impractical, unlike project1/2's
# tiny datasets) -- views only ever read them.
_cache = {}


def _load_json(path, key):
    if key not in _cache:
        if not os.path.exists(path):
            raise Http404("project3 artifacts not built yet -- run `python manage.py build_project3`.")
        with open(path) as f:
            _cache[key] = json.load(f)
    return _cache[key]


def _load_joblib(path, key):
    if key not in _cache:
        if not os.path.exists(path):
            raise Http404("project3 artifacts not built yet -- run `python manage.py build_project3`.")
        _cache[key] = joblib.load(path)
    return _cache[key]


def _metrics():
    return _load_json(precompute.METRICS_PATH, 'metrics')


def _demo_samples():
    return _load_json(precompute.DEMO_SAMPLES_PATH, 'demo_samples')


def _expert_pool():
    return _load_json(precompute.EXPERT_POOL_PATH, 'expert_pool')


def _baseline():
    return _load_joblib(precompute.BASELINE_PATH, 'baseline')


def _defer():
    return _load_joblib(precompute.DEFER_PATH, 'defer')


def _run_demo(sample):
    baseline = _baseline()
    defer = _defer()

    proba = baseline['model'].predict_proba(baseline['vectorizer'].transform([sample['text']]))[0]
    pred_idx = int(proba.argmax())
    ai_prediction = CLASS_NAMES[pred_idx]
    ai_confidence = float(proba[pred_idx])

    defer_prob = float(defer_predict_proba(defer['vectorizer'], defer['model'], [sample['text']])[0])
    will_defer = defer_prob > defer['threshold']

    expert_prediction = None
    if will_defer:
        expert_idx = int(simulate_expert_batch(
            np.array([sample['true_label']]), np.array([sample['index']]))[0])
        expert_prediction = CLASS_NAMES[expert_idx]

    final_prediction = expert_prediction if will_defer else ai_prediction
    return {
        'sample': sample,
        'ai_prediction': ai_prediction,
        'ai_confidence': ai_confidence,
        'defer_probability': defer_prob,
        'defer_threshold': defer['threshold'],
        'will_defer': will_defer,
        'expert_prediction': expert_prediction,
        'final_prediction': final_prediction,
        'correct': final_prediction == sample['true_label_name'],
    }


def index(request):
    metrics = _metrics()
    demo_samples = _demo_samples()

    context = {
        'metrics': metrics,
        'demo_samples': demo_samples,
        'selected_demo_index': None,
        'demo_result': None,
    }

    if request.method == 'POST':
        demo_index = request.POST.get('demo_index')
        if demo_index is not None:
            demo_index = int(demo_index)
            context['selected_demo_index'] = demo_index
            sample = next((s for s in demo_samples if s['index'] == demo_index), None)
            if sample is not None:
                context['demo_result'] = _run_demo(sample)

    return render(request, 'project3/index.html', context)


def serve_plot(request, filename):
    # Plots live under project3/artifacts/plots/ (build output, gitignored), not under
    # project3/static/ -- {% static %} can't see them, so they're served through this
    # dedicated view instead, the same way project1/2 serve their generated plots via
    # MEDIA_URL rather than static files.
    path = os.path.join(precompute.PLOTS_DIR, filename)
    if not os.path.exists(path):
        raise Http404("Plot not built yet -- run `python manage.py build_project3`.")
    return FileResponse(open(path, 'rb'), content_type='image/png')


def download_report(request):
    if not os.path.exists(REPORT_PATH):
        raise Http404("Report not built yet -- run `python manage.py build_project3`.")
    return FileResponse(
        open(REPORT_PATH, 'rb'), as_attachment=True,
        filename='project3_report.pdf', content_type='application/pdf',
    )


def expert_session(request):
    pool = _expert_pool()
    labels = request.session.get('project3_labels', {})

    if request.method == 'POST':
        idx = request.POST.get('index')
        guess = request.POST.get('guess')
        if idx is not None and guess in CLASS_NAMES:
            labels[idx] = guess
            request.session['project3_labels'] = labels
            request.session.modified = True
        elif request.POST.get('action') == 'reset':
            labels = {}
            request.session['project3_labels'] = labels
            request.session.modified = True

    scored = []
    for item in pool:
        key = str(item['index'])
        if key in labels:
            scored.append({
                'true_name': item['true_label_name'],
                'guess': labels[key],
                'correct': labels[key] == item['true_label_name'],
            })

    next_item = next((item for item in pool if str(item['index']) not in labels), None)

    overall_accuracy = (sum(s['correct'] for s in scored) / len(scored)) if scored else None
    class_rows = []
    for name in CLASS_NAMES:
        rows = [s for s in scored if s['true_name'] == name]
        your_acc = (sum(r['correct'] for r in rows) / len(rows)) if rows else None
        class_rows.append({'name': name, 'your_accuracy': your_acc, 'true_competence': COMPETENCE[name]})

    context = {
        'next_item': next_item,
        'num_labeled': len(scored),
        'pool_size': len(pool),
        'overall_accuracy': overall_accuracy,
        'class_rows': class_rows,
        'class_names': CLASS_NAMES,
        'done': next_item is None,
    }
    return render(request, 'project3/expert_session.html', context)

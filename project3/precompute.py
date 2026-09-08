"""One-shot orchestration for project3: trains/evaluates every model, writes all
artifacts consumed by the views/templates, and builds the PDF report (project3/report.py)
under project3/artifacts/. Run via `python manage.py build_project3`.
"""
import json
import os
import random
import time

import joblib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project3.data import get_train_data, get_test_data, CLASS_NAMES
from project3.baseline import train_baseline, evaluate_baseline
from project3.expert import evaluate_expert, simulate_expert_batch, COMPETENCE
from project3.defer import (
    compute_defer_targets, train_defer_model, defer_predict_proba,
    system_accuracy_at_threshold, oracle_accuracy, coverage_accuracy_curve, best_threshold,
)
from project3.active_learning import run_active_learning, compute_confidence, uncertainty_order
from project3 import report


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, 'plots')

BASELINE_PATH = os.path.join(ARTIFACTS_DIR, 'baseline.joblib')
DEFER_PATH = os.path.join(ARTIFACTS_DIR, 'defer.joblib')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')
DEMO_SAMPLES_PATH = os.path.join(ARTIFACTS_DIR, 'demo_samples.json')
EXPERT_POOL_PATH = os.path.join(ARTIFACTS_DIR, 'expert_session_pool.json')

COVERAGE_PLOT_PATH = os.path.join(PLOTS_DIR, 'coverage_curve.png')
AL_ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, 'active_learning_accuracy.png')
AL_DEFER_DENSITY_PLOT_PATH = os.path.join(PLOTS_DIR, 'active_learning_defer_density.png')
AL_COMPETENCE_MAE_PLOT_PATH = os.path.join(PLOTS_DIR, 'active_learning_competence_mae.png')
EXPERT_PER_CLASS_PLOT_PATH = os.path.join(PLOTS_DIR, 'expert_per_class.png')


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays (and dict/list containers) into plain
    Python types so json.dump doesn't choke on np.float64 / np.int64 / np.ndarray."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _render_coverage_plot(curve, best_t):
    thresholds = [row[0] for row in curve]
    accs = [row[1] for row in curve]
    rates = [row[2] for row in curve]
    best_acc = max(accs)
    for t, a, r in curve:
        if t == best_t:
            best_row_acc, best_row_rate = a, r
            break
    else:
        best_row_acc, best_row_rate = best_acc, None

    fig, ax = plt.subplots()
    ax.plot(thresholds, accs, marker='o', color='#1f77b4', label='System accuracy')
    ax.scatter([best_t], [best_row_acc], color='red', s=120, zorder=5,
               label=f'Best threshold ({best_t:.2f})')
    ax.set_xlabel('Defer threshold')
    ax.set_ylabel('System accuracy')

    ax2 = ax.twinx()
    ax2.plot(thresholds, rates, color='gray', linestyle='--', alpha=0.6, label='Deferral rate')
    ax2.set_ylabel('Deferral rate')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    ax.set_title('System accuracy & deferral rate vs. threshold')
    fig.tight_layout()
    fig.savefig(COVERAGE_PLOT_PATH)
    plt.close(fig)


def _render_al_accuracy_plot(al_results, ai_alone_acc, oracle_acc):
    # A plain auto-scaled y-axis is actively misleading here: system_accuracy barely moves
    # (~0.0002 total spread) at this query budget, so matplotlib's auto-scaling stretches
    # that noise to fill the whole plot and makes it look like dramatic swings. Anchoring
    # the y-axis to the AI-alone/oracle range shows the true (flat) picture honestly.
    fig, ax = plt.subplots()
    for strategy, color in (('uncertainty', '#1f77b4'), ('random', '#ff7f0e')):
        rounds = al_results[strategy]
        xs = [r['num_queries'] for r in rounds]
        ys = [r['system_accuracy'] for r in rounds]
        ax.plot(xs, ys, marker='o', markersize=3, color=color, label=strategy)
    ax.axhline(ai_alone_acc, color='gray', linestyle='--', linewidth=1, label=f'AI-alone ({ai_alone_acc:.3f})')
    ax.axhline(oracle_acc, color='green', linestyle='--', linewidth=1, label=f'Oracle ({oracle_acc:.3f})')
    ax.set_ylim(min(0.90, ai_alone_acc - 0.02), max(0.98, oracle_acc + 0.01))
    ax.set_xlabel('Number of queries')
    ax.set_ylabel('System accuracy')
    ax.set_title('Active learning: system accuracy vs. number of queries')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(AL_ACCURACY_PLOT_PATH)
    plt.close(fig)


def _render_al_line_plot(al_results, key, ylabel, title, path):
    fig, ax = plt.subplots()
    for strategy, color in (('uncertainty', '#1f77b4'), ('random', '#ff7f0e')):
        rounds = al_results[strategy]
        xs = [r['num_queries'] for r in rounds]
        ys = [r[key] for r in rounds]
        ax.plot(xs, ys, marker='o', markersize=3, color=color, label=strategy)
    ax.set_xlabel('Number of queries')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_al_competence_mae_plot(al_results):
    fig, ax = plt.subplots()
    for strategy, color in (('uncertainty', '#1f77b4'), ('random', '#ff7f0e')):
        rounds = al_results[strategy]
        xs = [r['num_queries'] for r in rounds if r['competence_mae'] is not None]
        ys = [r['competence_mae'] for r in rounds if r['competence_mae'] is not None]
        ax.plot(xs, ys, marker='o', markersize=3, color=color, label=strategy)
    ax.set_xlabel('Number of queries')
    ax.set_ylabel('Competence MAE')
    ax.set_title('Competence estimate error vs. number of queries')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(AL_COMPETENCE_MAE_PLOT_PATH)
    plt.close(fig)


def _render_expert_per_class_plot(per_class_acc, baseline_acc):
    names = CLASS_NAMES
    values = [per_class_acc[n] for n in names]

    fig, ax = plt.subplots()
    ax.bar(names, values, color='#1f77b4', label='Expert accuracy (per class)')
    ax.axhline(baseline_acc, color='red', linestyle='--', label=f'Baseline overall accuracy ({baseline_acc:.3f})')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Expert per-class accuracy vs. baseline overall accuracy')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(EXPERT_PER_CLASS_PLOT_PATH)
    plt.close(fig)


def run(budget=3000, batch_size=100):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()

    # ---- Task 1: baseline ----
    train_texts, train_labels = get_train_data()
    test_texts, test_labels = get_test_data()
    train_labels_arr = np.asarray(train_labels)
    test_labels_arr = np.asarray(test_labels)

    baseline_vectorizer, baseline_model = train_baseline(train_texts, train_labels)
    baseline_test_acc = evaluate_baseline(baseline_vectorizer, baseline_model, test_texts, test_labels)
    joblib.dump({'vectorizer': baseline_vectorizer, 'model': baseline_model}, BASELINE_PATH)

    # ---- Task 2: expert simulation ----
    test_idx = np.arange(len(test_labels_arr))
    expert_overall_acc, expert_per_class_acc = evaluate_expert(test_labels_arr, test_idx)

    # ---- Task 3: learning-to-defer ----
    ai_pred_train = baseline_model.predict(baseline_vectorizer.transform(train_texts))
    train_idx = np.arange(len(train_labels_arr))
    expert_pred_train = simulate_expert_batch(train_labels_arr, train_idx)
    defer_targets_train = compute_defer_targets(ai_pred_train, expert_pred_train, train_labels_arr)
    defer_vectorizer, defer_model = train_defer_model(train_texts, defer_targets_train)

    ai_pred_test = baseline_model.predict(baseline_vectorizer.transform(test_texts))
    expert_pred_test = simulate_expert_batch(test_labels_arr, test_idx)
    defer_prob_test = defer_predict_proba(defer_vectorizer, defer_model, test_texts)

    curve = coverage_accuracy_curve(ai_pred_test, expert_pred_test, test_labels_arr, defer_prob_test)
    best_t = best_threshold(curve)
    system_acc_at_best, deferral_rate_at_best = system_accuracy_at_threshold(
        ai_pred_test, expert_pred_test, test_labels_arr, defer_prob_test, best_t)
    ai_alone_acc = system_accuracy_at_threshold(
        ai_pred_test, expert_pred_test, test_labels_arr, defer_prob_test, 1.0)[0]  # threshold=1 -> never defer
    expert_alone_acc = float(np.mean(expert_pred_test == test_labels_arr))
    oracle_acc = oracle_accuracy(ai_pred_test, expert_pred_test, test_labels_arr)

    joblib.dump(
        {'vectorizer': defer_vectorizer, 'model': defer_model, 'threshold': float(best_t)},
        DEFER_PATH,
    )

    # ---- Task 4: active learning ----
    al_results = run_active_learning(
        baseline_vectorizer, baseline_model, train_texts, train_labels_arr,
        test_texts, test_labels_arr, batch_size=batch_size, budget=budget)

    # ---- Plots ----
    _render_coverage_plot(curve, best_t)
    _render_al_accuracy_plot(al_results, ai_alone_acc, oracle_acc)
    _render_al_line_plot(al_results, 'defer_positive_rate', 'Defer-positive rate',
                          'Active learning: defer-positive density vs. number of queries',
                          AL_DEFER_DENSITY_PLOT_PATH)
    _render_al_competence_mae_plot(al_results)
    _render_expert_per_class_plot(expert_per_class_acc, baseline_test_acc)

    # ---- metrics.json ----
    metrics = {
        'class_names': CLASS_NAMES,
        'baseline': {'test_accuracy': baseline_test_acc},
        'expert': {
            'overall_accuracy': expert_overall_acc,
            'per_class_accuracy': expert_per_class_acc,
            'competence_profile': {name: float(val) for name, val in COMPETENCE.items()},
        },
        'defer': {
            'ai_alone_accuracy': ai_alone_acc,
            'expert_alone_accuracy': expert_alone_acc,
            'oracle_accuracy': oracle_acc,
            'best_threshold': float(best_t),
            'system_accuracy_at_best': system_acc_at_best,
            'deferral_rate_at_best': deferral_rate_at_best,
            'coverage_curve': curve,
        },
        'active_learning': {
            'budget': budget,
            'batch_size': batch_size,
            'uncertainty': al_results['uncertainty'],
            'random': al_results['random'],
        },
    }
    metrics = _to_jsonable(metrics)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    # ---- demo_samples.json: 20 random TEST examples for the live-demo dropdown ----
    demo_idx = random.Random(42).sample(range(len(test_texts)), 20)
    demo_samples = [
        {
            'index': int(i),
            'text': test_texts[i],
            'true_label': int(test_labels_arr[i]),
            'true_label_name': CLASS_NAMES[int(test_labels_arr[i])],
        }
        for i in demo_idx
    ]
    with open(DEMO_SAMPLES_PATH, 'w') as f:
        json.dump(demo_samples, f, indent=2)

    # ---- expert_session_pool.json: 50 most-uncertain TRAIN examples for the
    # "you are the expert" interactive UI. NOTE: the view/template that serves this pool
    # to a live user MUST strip/hide `true_label`/`true_label_name` before showing the
    # article to the user -- they're only here so the server can score the guess after
    # the user submits it. ----
    confidence = compute_confidence(baseline_vectorizer, baseline_model, train_texts)
    most_uncertain_idx = uncertainty_order(confidence)[:50]
    expert_pool = [
        {
            'index': int(i),
            'text': train_texts[i],
            'true_label': int(train_labels_arr[i]),
            'true_label_name': CLASS_NAMES[int(train_labels_arr[i])],
        }
        for i in most_uncertain_idx
    ]
    with open(EXPERT_POOL_PATH, 'w') as f:
        json.dump(expert_pool, f, indent=2)

    # ---- PDF report (Task 3 PDF requirement) ----
    report.build_report(metrics)

    elapsed = time.time() - t0
    return {
        'baseline_test_accuracy': baseline_test_acc,
        'expert_overall_accuracy': expert_overall_acc,
        'system_accuracy_at_best': system_acc_at_best,
        'oracle_accuracy': oracle_acc,
        'best_threshold': float(best_t),
        'deferral_rate_at_best': deferral_rate_at_best,
        'elapsed_seconds': elapsed,
    }

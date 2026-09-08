import os
import random

import numpy as np

from django.contrib.admin.views.decorators import staff_member_required
from django.http import FileResponse, Http404
from django.shortcuts import render, redirect
from django.utils import timezone

from project4 import data
from project4.csv_export import write_responses_csv
from project4.models import StudySession, StudyResponse
from project4.preference_model import fit_preference_vector, predict_utility


DESIGN_CONFIG = {
    '1': {'rounds': 15, 'movies_per_round': 2, 'label': 'Pairwise comparison'},
    '2': {'rounds': 4, 'movies_per_round': 10, 'label': 'Top-10 ranking'},
}
VALIDATION_ROUNDS = 3
REG = 1.0

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
REPORT_PATH = os.path.join(ARTIFACTS_DIR, 'report.pdf')


def _movie_context(movies, i):
    row = movies.iloc[i]
    return {
        'index': int(i),
        'title': row['movie_title'],
        'genres': row['genres'].replace('|', ', '),
        'year': int(row['title_year']) if not np.isnan(row['title_year']) else None,
        'duration': row['duration'],
        'imdb_score': row['imdb_score'],
        'director': row['director_name'],
        'content_rating': row['content_rating'],
    }


def _sample_new(state, n):
    movies, X, _ = data.build_dataset()
    rng = np.random.default_rng()
    idx = data.sample_indices(n, rng, exclude=state.get('seen', []))
    state['seen'] = state.get('seen', []) + [int(i) for i in idx]
    return [int(i) for i in idx]


def index(request):
    return render(request, 'project4/index.html', {})


def preview(request):
    """Direct links to each design, bypassing random assignment. Deliberately not
    linked from the participant-facing landing page -- for the researcher's own use
    while setting up/checking the study, not something a real participant should see."""
    return render(request, 'project4/preview.html', {})


@staff_member_required
def export_data(request):
    return write_responses_csv(StudyResponse.objects.all())


def start_study(request):
    design = random.choice(list(DESIGN_CONFIG.keys()))
    return redirect('project4:study', design=design)


def study(request, design):
    if design not in DESIGN_CONFIG:
        raise Http404("Unknown design")
    config = DESIGN_CONFIG[design]

    state = request.session.get('project4_study')
    if state is None or state.get('design') != design:
        session_obj = StudySession.objects.create(design=design)
        state = {'design': design, 'phase': 'elicit', 'responses': [], 'validation': [],
                  'pending': [], 'seen': [], 'session_id': session_obj.id}

    if request.method == 'POST':
        if state['phase'] == 'elicit':
            pending = state['pending']
            if design == '1':
                winner = request.POST.get('winner')
                if winner is not None and int(winner) in pending:
                    winner = int(winner)
                    loser = [m for m in pending if m != winner][0]
                    state['responses'].append([winner, loser])
                    StudyResponse.objects.create(
                        session_id=state['session_id'], round_number=len(state['responses']),
                        phase='elicit', movies_shown=pending, winner_index=winner,
                    )
                    state['pending'] = []
            else:
                try:
                    positions = {m: int(request.POST.get(f'rank_{m}')) for m in pending}
                except (TypeError, ValueError):
                    positions = {}
                if sorted(positions.values()) == list(range(1, len(pending) + 1)):
                    ordered = sorted(pending, key=lambda m: positions[m])
                    state['responses'].append(ordered)
                    StudyResponse.objects.create(
                        session_id=state['session_id'], round_number=len(state['responses']),
                        phase='elicit', movies_shown=pending, ranking_order=ordered,
                    )
                    state['pending'] = []
            if len(state['responses']) >= config['rounds']:
                state['phase'] = 'validate'
        elif state['phase'] == 'validate':
            pending = state['pending']
            winner = request.POST.get('winner')
            if winner is not None and int(winner) in pending:
                winner = int(winner)
                loser = [m for m in pending if m != winner][0]
                state['validation'].append([winner, loser])
                StudyResponse.objects.create(
                    session_id=state['session_id'], round_number=len(state['validation']),
                    phase='validate', movies_shown=pending, winner_index=winner,
                )
                state['pending'] = []
            if len(state['validation']) >= VALIDATION_ROUNDS:
                state['phase'] = 'done'

    if state['phase'] == 'elicit' and not state['pending']:
        state['pending'] = _sample_new(state, config['movies_per_round'])
    elif state['phase'] == 'validate' and not state['pending']:
        state['pending'] = _sample_new(state, 2)

    movies, X, feature_names = data.build_dataset()

    if state['phase'] == 'done':
        return _render_results(request, state, movies, X, feature_names, config)

    request.session['project4_study'] = state
    request.session.modified = True

    is_validation = state['phase'] == 'validate'
    context = {
        'design': design,
        'config': config,
        'is_validation': is_validation,
        'round_number': (len(state['validation']) if is_validation else len(state['responses'])) + 1,
        'total_rounds': VALIDATION_ROUNDS if is_validation else config['rounds'],
        'pending_movies': [_movie_context(movies, i) for i in state['pending']],
        'positions': list(range(1, config['movies_per_round'] + 1)),
    }
    return render(request, 'project4/study.html', context)


def _render_results(request, state, movies, X, feature_names, config):
    rankings = [X[idx_list] for idx_list in state['responses']]
    w = fit_preference_vector(rankings, n_features=X.shape[1], reg=REG)

    correct = 0
    for winner, loser in state['validation']:
        if predict_utility(w, X[winner]) > predict_utility(w, X[loser]):
            correct += 1
    validation_accuracy = correct / len(state['validation']) if state['validation'] else None

    utilities = predict_utility(w, X)
    seen = set(state.get('seen', []))
    candidates = [i for i in range(len(movies)) if i not in seen]
    top_idx = sorted(candidates, key=lambda i: -utilities[i])[:5]

    top_features = sorted(zip(feature_names, w), key=lambda kv: -abs(kv[1]))[:8]

    StudySession.objects.filter(id=state['session_id']).update(
        completed_at=timezone.now(),
        validation_accuracy=validation_accuracy,
        fitted_weights=[float(v) for v in w],
    )

    # Only the transient UI state is cleared here -- the StudySession/StudyResponse rows
    # created above are never deleted, so the raw data survives after this screen renders.
    del request.session['project4_study']
    request.session.modified = True

    context = {
        'design': state['design'],
        'config': config,
        'num_responses': len(state['responses']),
        'validation_accuracy': validation_accuracy,
        'num_validation': len(state['validation']),
        'top_features': top_features,
        'top_movies': [_movie_context(movies, i) for i in top_idx],
    }
    return render(request, 'project4/results.html', context)


def download_report(request):
    if not os.path.exists(REPORT_PATH):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        from project4 import report
        report.build_report()
    return FileResponse(
        open(REPORT_PATH, 'rb'), as_attachment=True,
        filename='project4_report.pdf', content_type='application/pdf',
    )

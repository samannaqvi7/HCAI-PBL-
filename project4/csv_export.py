import csv

from django.http import HttpResponse


FIELDNAMES = [
    'session_id', 'design', 'session_started_at', 'session_completed_at',
    'validation_accuracy', 'fitted_weights',
    'round_number', 'phase', 'movies_shown', 'winner_index', 'ranking_order',
    'response_created_at',
]


def write_responses_csv(queryset, filename='project4_study_data.csv'):
    """One row per StudyResponse, with its parent session's fields repeated -- a
    denormalized 'long' format so the whole dataset comes out as a single flat file,
    joinable/filterable directly without needing to reassemble sessions and responses."""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    writer = csv.writer(response)
    writer.writerow(FIELDNAMES)
    for r in queryset.select_related('session').order_by('session_id', 'phase', 'round_number'):
        s = r.session
        writer.writerow([
            s.id, s.design, s.started_at.isoformat(),
            s.completed_at.isoformat() if s.completed_at else '',
            s.validation_accuracy, s.fitted_weights,
            r.round_number, r.phase, r.movies_shown, r.winner_index, r.ranking_order,
            r.created_at.isoformat(),
        ])
    return response

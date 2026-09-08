from django.db import models


class StudySession(models.Model):
    """One participant's full run through either design, from first round to fitted
    result. Created the moment a participant enters `study()`, updated in place as
    responses come in, never deleted when the session finishes -- only the transient
    request.session UI state is cleared then."""

    DESIGN_CHOICES = [('1', 'Design 1 (pairwise)'), ('2', 'Design 2 (ranking)')]

    design = models.CharField(max_length=1, choices=DESIGN_CHOICES)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    fitted_weights = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Session {self.id} (Design {self.design})"


class StudyResponse(models.Model):
    """One elicitation or validation round's raw response. Exactly one of
    winner_index (pairwise: Design 1's rounds and every design's validation rounds) or
    ranking_order (Design 2's rounds: full best-to-worst movie index order) is set."""

    PHASE_CHOICES = [('elicit', 'Elicitation'), ('validate', 'Validation')]

    session = models.ForeignKey(StudySession, on_delete=models.CASCADE, related_name='responses')
    round_number = models.PositiveIntegerField()
    phase = models.CharField(max_length=10, choices=PHASE_CHOICES)
    movies_shown = models.JSONField()
    winner_index = models.IntegerField(null=True, blank=True)
    ranking_order = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['session', 'phase', 'round_number']

    def __str__(self):
        return f"Session {self.session_id} round {self.round_number} ({self.phase})"

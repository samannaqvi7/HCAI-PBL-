from django.contrib import admin

from .csv_export import write_responses_csv
from .models import StudySession, StudyResponse


def export_selected_responses(modeladmin, request, queryset):
    responses = StudyResponse.objects.filter(session__in=queryset)
    return write_responses_csv(responses, filename='project4_selected_sessions.csv')


export_selected_responses.short_description = "Export selected sessions' responses to CSV"


class StudyResponseInline(admin.TabularInline):
    model = StudyResponse
    extra = 0
    readonly_fields = ('round_number', 'phase', 'movies_shown', 'winner_index', 'ranking_order', 'created_at')
    can_delete = False


@admin.register(StudySession)
class StudySessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'design', 'started_at', 'completed_at', 'validation_accuracy')
    list_filter = ('design',)
    inlines = [StudyResponseInline]
    actions = [export_selected_responses]


@admin.register(StudyResponse)
class StudyResponseAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'round_number', 'phase', 'winner_index', 'created_at')
    list_filter = ('phase',)

from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('report.pdf', views.download_report, name='download_report'),
    path('expert-session', views.expert_session, name='expert_session'),
    path('plots/<str:filename>', views.serve_plot, name='plot'),
]

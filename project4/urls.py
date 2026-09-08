from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.index, name='index'),
    path('report.pdf', views.download_report, name='download_report'),
    path('start', views.start_study, name='start_study'),
    path('study/<str:design>', views.study, name='study'),
    path('preview', views.preview, name='preview'),
    path('export', views.export_data, name='export_data'),
]

# from django.conf.urls import url
from django.urls import re_path as url

from severity import views

urlpatterns = [
    url('predict', views.PredictionDetails.as_view()),
]

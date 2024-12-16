# product_finder/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload'),
    path('results/<int:search_id>/', views.results, name='results'),
]
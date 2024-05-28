from django.urls import path
from . import views

app_name = 'restaurant'
#추후 어플 이름 변경 예정

urlpatterns = [
    path('restaurant/', views.restaurant_list, name='restaurant_list'),
    path('restaurant/<int:restaurant_id>/', views.restaurant_detail, name='restaurant_detail'),
]
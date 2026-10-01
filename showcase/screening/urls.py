from django.urls import path

from screening import views

urlpatterns = [
    path("", views.home, name="home"),
    path("product", views.product, name="product"),
    path("about", views.about, name="about"),
    path("api/screen", views.screen, name="screen"),
    path("api/note", views.note, name="note"),
    path("api/chat", views.chat, name="chat"),
    path("health", views.health, name="health"),
]

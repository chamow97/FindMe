
from django.conf.urls import url

from findmeapp import views

app_name='twitter_snaps'

urlpatterns = [
    url(r'^$', views.index, name='index'),
]
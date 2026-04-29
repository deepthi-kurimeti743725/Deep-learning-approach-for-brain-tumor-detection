from django.urls import path
from . import views
urlpatterns = [
    path('', views.SigUp , name='userlogin') ,
    path('UserLogin/' , views.UserLogin , name='UserLogin'),
    path('User_Home/' , views.UserHome , name='UserHome'),
    path('Traning/' , views.Traning , name='Traning'),
    path('predict/' , views.predict , name='predict')

]
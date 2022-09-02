from django.urls import path
from app import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.sign_up, name='signup'),
    path('signin/', views.login_page, name='signin'),
    path('signout/', views.logout_page, name='signout'),
    path('upload/', views.upload, name='upload'),
    path('eda/', views.eda, name='eda'),
    path('data_preprocessing/', views.data_preprocessing, name='data_preprocessing'),
    path('model_selection/', views.model_selection, name='model_selection'),
    path('model_evaluation/', views.model_evaluation, name='model_evaluation'),
    path('profile_data/', views.profile_data, name='profile_data'),
    path('model_testing/<button_id>', views.model_testing, name='model_testing'),
    path('save_model/', views.save_model, name='save_model'),
    # path('profile/', views.profile, name='profile'),
]

from app import views
from django.urls import path

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.sign_up, name='signup'),
    path('continue_as_guest/', views.continue_as_guest, name='continue_as_guest'),
    path('signin/', views.login_page, name='signin'),
    path('signout/', views.logout_page, name='signout'),
    path('upload/', views.upload, name='upload'),
    path('eda/', views.eda, name='eda'),
    path('advanced_eda/', views.advanced_eda, name='advanced_eda'),
    path('data_preprocessing/', views.data_preprocessing, name='data_preprocessing'),
    path('model_selection/', views.model_selection, name='model_selection'),
    path('model_evaluation/', views.model_evaluation, name='model_evaluation'),
    path('profile_data/', views.profile_data, name='profile_data'),
    path('model_testing/<button_id>', views.model_testing, name='model_testing'),
    path('save_model/', views.save_model, name='save_model'),
    # path('profile/', views.profile, name='profile'),

    path('api.model_test_api/', views.model_test_api),
    path('api.delete_model_api/', views.delete_model_api)
]

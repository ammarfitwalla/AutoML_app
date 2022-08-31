import os
import pickle
import chardet
from .utils import *
import pandas as pd
import numpy as np
from app.models import *
from sklearn import metrics
import matplotlib.pyplot as plt
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.utils.text import get_valid_filename
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from django.core.files.storage import FileSystemStorage
from sklearn.metrics._classification import _check_targets
from django.contrib.auth import authenticate, login, logout, decorators
from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from pandas.api.types import is_numeric_dtype, is_float_dtype, is_string_dtype
import category_encoders as ce

val = None
model_value = None
model = None
predictions = None
used_model = []
categorical_model_name_list = ['Logistic Regression', 'Decision Tree Classifier', 'KNeighbors Classifier',
                               'Random Forest Classifier', 'GaussianNB Classifier', 'SGD Classifier']
numerical_model_name_list = ['Linear Regression']
media_path = settings.MEDIA_ROOT

sc_X = StandardScaler()

plt.switch_backend('agg')


def home(request):
    return render(request, 'home.html')


def sign_up(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # ============== VALIDATIONS ============== #

        # USERNAME
        if User.objects.filter(username=username).exists():
            messages.warning(request, 'Username already present, please choose different username')
            return redirect('/signup/')

        if len(username) > 10:
            messages.warning(request, "username must be below 10 characters")
            return redirect('/signup/')

        if not username.isalnum():
            messages.warning(request, "username must be Alphanumeric only")
            return redirect('/signup/')

        # EMAIL
        if User.objects.filter(email=email).exists():
            messages.warning(request, "Email already present")
            return redirect('/signup/')

        user = User.objects.create_user(username, email, password)
        user.save()

        custom_user = Profile(
            user=user
        )
        custom_user.save()
        messages.success(request, 'account created, please login here')
        return redirect('/signin/')

    return render(request, 'signup.html')


def login_page(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'Successfully Logged in')
            return redirect('/')
        else:
            messages.error(request, 'Invalid Credentials')
            return redirect('/signin/')
    return render(request, 'login.html')


def logout_page(request):
    logout(request)
    messages.success(request, "Successfully logged out")
    return redirect('/signin/')


@decorators.login_required
@decorators.login_required
def upload(request):
    global uploaded_file_url
    if not request.user.is_authenticated:
        return redirect('login')

    elif request.method == 'POST':
        user = str(request.user.id)
        if not os.path.isdir(os.path.join(media_path, user)):
            os.mkdir(os.path.join(media_path, user))

        myFile = request.FILES.get('myFile')
        if os.path.splitext(myFile.name)[-1] == ".csv":
            # fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'documents/user_' + user),
            #                        base_url='documents/')
            fs = FileSystemStorage(location=os.path.join(media_path, user, 'documents' + os.sep),
                                   base_url=media_path + os.sep + 'documents' + os.sep)
            myFile.name = get_valid_filename(myFile.name)
            filename = fs.save(myFile.name, myFile)
            uploaded_file_url = fs.url(filename)
            messages.success(request, 'File Uploaded')

            return redirect('/eda/')
        else:
            messages.error(request, 'Please select valid file with extension .csv')
            return redirect('/upload/')

    return render(request, 'upload.html')


def eda(request):
    user = request.user.id
    print('USER', user)

    if user is not None:
        all_data = Profile.objects.filter(user_id=user).values('document')

        # print(all_data)
        if len(all_data) > 0:
            file_name = os.path.join(settings.MEDIA_ROOT, all_data[0]['document'])
            with open(file_name, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(10000))
            # print(result)

            df = pd.read_csv(file_name, encoding=result['encoding'])
            df_html = df.head().to_html(classes="table table-striped")
            df_n_rows = df.shape[0]
            df_n_cols = df.shape[1]
            df_cols = [column for column in df.columns]
            df_null = df.isnull().values.any()
            df_describe = df.describe()
            df_describe_html = df_describe.to_html(classes="table table-striped")
            categorical = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique() < df.shape[0] // 2]
            # categorical = [col for col in df.columns if df[col].nunique() > df.shape[0]/2]

            png_files_path = []
            folder = os.path.join(settings.MEDIA_ROOT, 'graphs')
            for i in categorical:
                # df[i].value_counts().plot(kind='bar')
                png_file_name = folder + os.sep + i + ".png"
                sns.countplot(x=df[i], data=df)
                plt.savefig(png_file_name)
                png_files_path.append(png_file_name)
            print(png_files_path)

            if request.method == 'POST':
                dependent_variable = request.POST['dep_var_name']
                slider_value = request.POST['slider_value']
                test_size_ratio = (100 - int(slider_value)) / 100
                # print("test_size_ratio:", test_size_ratio)
                selected_check_list = request.POST.getlist('checkbox_name')
                # print("selected_check_list: ", type(selected_check_list))
                if dependent_variable not in selected_check_list and df_n_cols - len(selected_check_list) > 1:
                    df = df.drop(selected_check_list, axis=1)
                    global val

                    def val():
                        return [df, dependent_variable, str(df.dtypes[dependent_variable]), test_size_ratio]

                    return redirect('/model_selection/')
                elif df_n_cols - len(selected_check_list) <= 1:
                    messages.error(request, 'You cannot delete all the columns')
                else:
                    messages.error(request, 'You cannot delete prediction column')

                return redirect('/eda/')

            context = {
                'df_html': df_html,
                'df_n_rows': df_n_rows,
                'df_n_cols': df_n_cols,
                'df_cols': df_cols,
                'df_null': df_null,
                'df_describe_html': df_describe_html,
                'png_files_path': png_files_path,
            }
            return render(request, "eda.html", context)
        else:
            return redirect('/upload/')
    else:
        return redirect('/signin/')


def data_preprocessing(request):
    global eda_val
    eda_values = eda_val()
    df_preprocessing = eda_values[0]
    df_col_numbers = eda_values[1]
    df_col_names = eda_values[2]
    categorical_col_names = eda_values[3]
    df_null = df_preprocessing.isnull().values.any()
    df_null_columns = None
    all_null_columns = None
    string_cols = None
    list_to_handle_nan_values = ['mean', 'median', 'bfill', 'ffill', 0, 'delete records']
    list_to_handle_nan_str_values = ['bfill', 'ffill', 0, 'delete records']

    if df_null:
        df_null_columns, string_cols = [], []
        all_null_columns = df_preprocessing.columns[df_preprocessing.isna().any()].tolist()

        for i in all_null_columns:
            if is_string_dtype(df_preprocessing[i]):
                string_cols.append(i)
            else:
                df_null_columns.append(i)

    if request.method == 'POST':
        dependent_variable = request.POST['dep_var_name']
        slider_value = request.POST['slider_value']
        test_size_ratio = (100 - int(slider_value)) / 100
        # print(df_preprocessing.shape)

        if df_null:
            print("=================================")
            for i in all_null_columns:
                way = request.POST.get(i)
                print(way)
                if way == '0':
                    df_preprocessing[i] = df_preprocessing[i].fillna(0)
                elif way == 'bfill' or way == 'ffill':
                    df_preprocessing[i] = df_preprocessing[i].fillna(method=way)
                elif way == 'delete records':
                    df_preprocessing.dropna(subset=[i], inplace=True)
                elif way == 'mean':
                    df_preprocessing[i] = df_preprocessing[i].fillna((df_preprocessing[i].mean()))
                elif way == 'median':
                    df_preprocessing[i] = df_preprocessing[i].fillna((df_preprocessing[i].median()))

        print("=================================")
        print(df_preprocessing.shape)
        df_preprocessing.to_csv('check_fixed_nan.csv')
        # print("test_size_ratio:", test_size_ratio)
        selected_check_list = request.POST.getlist('checkbox_name')
        # print("selected_check_list: ", type(selected_check_list))

        oh_encoder = ce.OrdinalEncoder(cols=categorical_col_names)
        df_preprocessing = oh_encoder.fit_transform(df_preprocessing)

        if dependent_variable not in selected_check_list and df_col_numbers - len(selected_check_list) > 1:
            df_preprocessing = df_preprocessing.drop(selected_check_list, axis=1)
            global val

            def val():
                return [df_preprocessing, dependent_variable, str(df_preprocessing.dtypes[dependent_variable]),
                        test_size_ratio]

            return redirect('/model_selection/')
        elif df_col_numbers - len(selected_check_list) <= 1:
            messages.error(request, 'You cannot delete all the columns')
        else:
            messages.error(request, 'You cannot delete prediction column')

    context = {
        'df_null': df_null,
        'nan_columns': df_null_columns,
        'string_cols': string_cols,
        'list_handle_nan_values': list_to_handle_nan_values,
        'list_handle_nan_str_values': list_to_handle_nan_str_values,
        'df_cols': df_col_names,
    }
    return render(request, 'data_preprocessing.html', context)


def model_selection(request):
    data = None
    lr = None
    if request.is_ajax():
        print('AJAX REQUEST')
        data = request.GET.get('data')
        print('DATA is:', data)
        lr = data
    # try:
    global model, predictions
    value = val()
    df_model = value[0]
    dependent_variable = value[1]
    dependent_variable_type = value[2]
    test_size_ratio = value[3]
    # print("dependent_variable_type: ", dependent_variable_type)

    # TODO : Display model based on dep var type

    X = df_model.drop([dependent_variable], axis=1)
    y = df_model[dependent_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size_ratio,
                                                        random_state=77,
                                                        shuffle=True)

    # m_name = request.GET.get('model_name')
    # print("m_name", m_name)
    if data == 'Linear Regression':
        lr = 'Linear Regression'
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        button_name = request.POST.get('evaluation')
        # print("model_name", model_name)
        selected_model = request.POST.get('selected_model')

        if button_name is None:

            if model_name == 'Linear Regression':
                model = get_linear_regression_model(X_train, y_train)
                predictions = model.predict(X_test)
            else:
                p_X_train = X_train.copy()
                p_X_test = X_test.copy()
                print(X_test)
                print(X_test.shape)
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                if model_name == 'Logistic Regression':
                    model = get_logistic_regression_model(X_train, y_train)
                elif model_name == 'Decision Tree Classifier':
                    model = get_decision_tree_classifier_model(X_train, y_train)
                elif model_name == 'Random Forest Classifier':
                    model = get_random_forest_classifier_model(X_train, y_train)
                elif model_name == 'GaussianNB':
                    model = get_gaussian_nb_model(X_train, y_train)
                elif model_name == 'SGDClassifier':
                    model = get_sgd_classifier_model(X_train, y_train)
                else:
                    model = get_kneighbors_classifier_model(X_train, y_train)

                predictions = model.predict(X_test)
                X_train = p_X_train
                X_test = p_X_test

            actual_pred_df = X_test.copy()
            actual_pred_df['Actual output'] = y_test
            actual_pred_df['Predicted output'] = predictions
            actual_pred_df = actual_pred_df.to_html(classes="table table-striped table-hover")
            params = None

            context = {
                'lr': lr,
                'params': params,
                'actual_pred_df': actual_pred_df,
                'model_name_list': categorical_model_name_list + numerical_model_name_list
            }

            used_model.append(model_name)

            return render(request, 'model_selection.html', context)
        else:
            print("used_model", used_model)
            global model_value

            def model_value():
                return [used_model[-1], model, X_train, X_test, y_train, y_test, predictions, df_model, X, y]

            return redirect('/model_evaluation/')

    context = {
        'model_name_list': categorical_model_name_list + numerical_model_name_list,
        'lr': lr
    }

    return render(request, 'model_selection.html', context)
    # except:
    #     return redirect('/eda/')


def model_evaluation(request):
    # try:
    value = model_value()
    model_name = value[0]
    model = value[1]
    X_train = value[2]
    X_test = value[3]
    y_train = value[4]
    y_test = value[5]
    y_pred = value[6]
    print(y_pred)

    if model_name == 'Linear Regression':
        folder_ = os.path.join(settings.MEDIA_ROOT, 'regression_plot')
        png_file_name_ = folder_ + os.sep + "true_vs_predictions.png"
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        model_score = '%.2f' % (model.score(X_test, y_test) * 100) + ' %'
        my_data = [['Model', model_name], ['Mean Absolute Error', mae], ['Mean Squared Error', mse],
                   ['Root Mean Squared Error', rmse], ['Model Score', model_score]]

        plt.scatter(y_test, y_pred, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(y_pred), max(y_test))
        p2 = min(min(y_pred), min(y_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.savefig(png_file_name_)

    else:
        # print(model_name)
        # cm = confusion_matrix(y_test, y_pred)
        # print("TN", cm[0][0], "FP", cm[0][1])
        # print("FN", cm[1][0], "TP", cm[1][1])
        # class_report = metrics.classification_report(y_test, y_pred)
        # print("class_report: ", class_report)

        if _check_targets(y_test, y_pred)[0] == 'multiclass':
            average = 'multiclass'
        else:
            average = 'binary'

        accuracy = '%.2f' % (metrics.accuracy_score(y_test, y_pred) * 100) + ' %'
        recall = '%.2f' % (metrics.recall_score(y_test, y_pred, average=average) * 100) + ' %'
        f1 = '%.2f' % (metrics.f1_score(y_test, y_pred, average=average) * 100) + ' %'
        precision = '%.2f' % (metrics.precision_score(y_test, y_pred, average=average) * 100) + ' %'

        my_data = [['Model', model_name], ['Accuracy Score', accuracy], ['Recall Score', recall], ['F1 score', f1],
                   ['Precision', precision]]

        png_file_name_ = None

    df_ev = pd.DataFrame(my_data, columns=['Metrics', 'Values'])
    df_to_html = df_ev.to_html(classes="table table-striped table-hover")

    context = {
        'model_name': model_name,
        'df': df_to_html,
        'fig': png_file_name_,
    }
    return render(request, 'model_evaluation.html', context)


# except:
#     return redirect('/model_selection/')

def profile_data(request):
    user = str(request.user.id)
    print(user)
    if user:
        context = None
        document = Profile.objects.filter(user_id=user).values('document')
        if document:
            document = document[0]['document'].split("/")[-1]

        model_data = TrainedModels.objects.filter(user_id=user).values()
        if model_data:
            project_name = model_data[0]['project_name']
            if request.method == 'POST':
                return redirect('/model_testing/')

            context = {
                'document': document,
                'project_name': project_name
            }

        return render(request, 'profile_data.html', context)
    else:
        return redirect('/signin/')


def model_testing(request):
    # test_model = model_value()
    # df = test_model[7]
    # df_cols = [column for column in df.columns]
    # if request.method == 'POST':
    #     cols_values = request.POST.getlist('cols_values')
    #
    # context = {
    #     'cols': df_cols
    #
    # }
    if request.user.id:
        user = str(request.user.id)
        model_data = TrainedModels.objects.filter(user_id=user).values()
        custom_predictions = None
        df_test = None
        model_file = model_data[0]['model_file']
        project_name = model_data[0]['project_name']
        model_name = model_data[0]['model_name']
        col_names = model_data[0]['column_names']
        col_names = list(col_names[1:-1])
        col_names = "".join(col_names)
        col_names = col_names.split(", ")
        col_names = [i[1:-1] for i in col_names]
        if request.method == 'POST':
            predict = request.POST.getlist('inputs')
            predict = [[int(i) for i in predict]]
            print(predict)
            model_file = pickle.loads(model_file)
            if model_name in categorical_model_name_list:
                print(col_names)
                predict = pd.DataFrame(predict, columns=col_names)
                print(predict)
                print('Model Name:', model_name)
                predict = sc_X.transform(predict)
                # predict = sc_X.fit_transform([predict])
                print(predict)
                custom_predictions = model_file.predict(predict)
                custom_predictions = str(custom_predictions[0])
            else:
                custom_predictions = model_file.predict([predict])
                custom_predictions = str(custom_predictions[0])

            test_data = [['Prediction', custom_predictions]]
            print("custom_predictions", custom_predictions)
            print("predict", predict)
            df_test = pd.DataFrame(test_data)
            df_test = df_test.to_html(classes="table table-striped table-hover")
        context = {
            'model_name': model_name,
            'predictions': df_test,
            'col': col_names,
            'project_name': project_name
        }
        # print(context)

        return render(request, 'model_testing.html', context)
    else:
        return redirect('/signin/')


def save_model(request):
    user = request.user.id
    user = User.objects.get(id=user)

    model_attributes = model_value()
    model = model_attributes[1]
    model_name = model_attributes[0]
    X = model_attributes[8]
    y = model_attributes[9]

    X_cols = [column for column in X.columns]
    if request.method == 'POST':
        project_name = request.POST.get('project_name')
        pickle_file = pickle.dumps(model)

        model_data = TrainedModels.objects.filter(user_id=user)

        if not model_data:
            data = TrainedModels(
                user=user,
                project_name=project_name,
                model_file=pickle_file,
                column_names=X_cols,
                model_name=model_name,
            )
            data.save()
            messages.success(request, 'Your first Project has been saved successfully !')
        else:
            var = TrainedModels.objects.filter(user_id=user).update(project_name=project_name,
                                                                    model_file=pickle_file,
                                                                    column_names=X_cols,
                                                                    model_name=model_name,
                                                                    )
            messages.success(request, 'Your Project has been updated successfully !')

        return redirect('/profile_data/')
    return render(request, 'save_model.html')

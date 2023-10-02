import os
import csv
import base64
import requests
import chardet
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from flaml import AutoML
from json import JSONEncoder
from datetime import datetime
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier


media_path = settings.MEDIA_ROOT
automl = AutoML()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def check_dir_exists(path):
    """
    :param path: /path/to/dir
    :return: if dir not present, it creates a dir
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def divide_columns_into_df(df):
    num_columns = len(df.columns)
    if num_columns < 8:
        return [df]
    num_df = 2

    columns_per_df = num_columns // num_df
    remainder = num_columns % num_df

    dfs = []

    start_col = 0
    for i in range(num_df):
        end_col = start_col + columns_per_df
        if i < remainder:
            end_col += 1

        subset_df = df.iloc[:, start_col:end_col]
        dfs.append(subset_df)
        start_col = end_col

    return dfs


def file_to_df(file_name):
    with open(file_name, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))

    with open(file_name, newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read())
        if dialect.delimiter == ',':
            df = pd.read_csv(file_name, encoding=result['encoding'])  # Import the csv with a comma as the separator
        elif dialect.delimiter == ';':
            df = pd.read_csv(file_name, sep=';', encoding=result['encoding'])  # Import the csv with a semicolon as the separator
        return df


def get_automl_model(task_type, X_train, y_train):
    print('task_type', task_type)
    automl.fit(X_train, y_train, task=str(task_type), time_budget=5, early_stop=True)
    best_model = automl.best_estimator
    return best_model, automl

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_png = buffer.getvalue()
    graph = base64.b64encode(img_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def get_plot(x, y, title, xname, yname):
    graph = get_graph()
    plt.switch_backend('AGG')
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.plot(x, y)
    plt.xticks(rotation=45)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.tight_layout()

    return graph

def get_cat_plot(x, df):
    graph = get_graph()
    plt.switch_backend('AGG')
    plt.figure(figsize=(8, 5))
    plt.plot(df[x].value_counts())
    # sns.countplot(x=x, data=df, hue=hue)
    plt.tight_layout()

    return graph


def find_outliers_limit(df, col):
    print(col)
    print('-' * 50)
    # removing outliers
    q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    print('Lower:', lower, ' Upper:', upper)
    return lower, upper


def remove_outlier(df, col, upper, lower):
    # identify outliers
    outliers = [x for x in df[col] if x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in df[col] if lower <= x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    final = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))
    return final


# outlier_cols = ['Levy', 'Engine volume', 'Mileage', 'Cylinders']
# for col in outlier_cols:
#     lower, upper = find_outliers_limit(df, col)
#     df[col] = remove_outlier(df, col, upper, lower)


def get_linear_regression_model(X_train, y_train):
    linear_regression_obj = LinearRegression()
    linear_regression_obj.fit(X_train, y_train)

    return linear_regression_obj


def get_logistic_regression_model(X_train, y_train):
    logistic_regression_obj = LogisticRegression()
    logistic_regression_obj.fit(X_train, y_train)

    return logistic_regression_obj


def get_decision_tree_classifier_model(X_train, y_train):
    decision_tree_classifier_obj = DecisionTreeClassifier()
    decision_tree_classifier_obj.fit(X_train, y_train)

    return decision_tree_classifier_obj


def get_sgd_classifier_model(X_train, y_train):
    sgd_obj = SGDClassifier()
    sgd_obj.fit(X_train, y_train)

    return sgd_obj


def get_kneighbors_classifier_model(X_train, y_train):
    kneighbors_classifier_obj = KNeighborsClassifier()
    kneighbors_classifier_obj.fit(X_train, y_train)

    return kneighbors_classifier_obj


def get_random_forest_classifier_model(X_train, y_train):
    random_forest_classifier_obj = RandomForestClassifier()
    random_forest_classifier_obj.fit(X_train, y_train)

    return random_forest_classifier_obj


def get_gaussian_nb_model(X_train, y_train):
    gaussian_nb_obj = GaussianNB()
    gaussian_nb_obj.fit(X_train, y_train)

    return gaussian_nb_obj


def get_user_info(user_ip):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    response = requests.get(f"https://ipinfo.io/{user_ip}/json")
    data = response.json()

    data_dict = {'City': data.get('city', 'Unknown'),
                 'Region': data.get('region', 'Unknown'),
                 'Country': data.get('country', 'Unknown'),
                 'Location': data.get('loc', 'Unknown'),
                 'Date': formatted_datetime}

    return data_dict

def write_user_info_to_excel(data_dict):

    csv_file_path = media_path + os.sep + 'data_with_datetime.csv'
    existing_data = pd.read_csv(csv_file_path)
    updated_data = pd.concat([existing_data, pd.DataFrame([data_dict])], ignore_index=True)
    updated_data.to_csv(csv_file_path, index=False)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

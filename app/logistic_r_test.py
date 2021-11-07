import os
import sys
import chardet
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import tpe
from flaml import AutoML
from sklearn import metrics
import matplotlib.pyplot as plt
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hpsklearn import HyperoptEstimator
from sklearn.metrics import confusion_matrix
from flaml.ml import sklearn_metric_loss_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import check_classification_targets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics._classification import _check_targets, accuracy_score

file_name = r'D:\KS\Projects\django_ml\media\documents\modified_sample_5H3J7pV.csv'
# print(os.extsep)
with open(file_name, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
# print(result)

df = pd.read_csv(file_name, encoding=result['encoding'])
# print(df.head())

# sns.pairplot(df)
# plt.show()

categorical = [var for var in df.columns if df[var].dtype == 'O']

# png_files_path = []
# for i in categorical:
#     # df[i].value_counts().plot(kind='bar')
#     png_file_name = i + ".png"
#     sns.countplot(df[i], data=df)
#     plt.savefig(png_file_name)
#     png_files_path.append(png_file_name)
# print(png_files_path)
# sys.exit()
# print('There are {} categorical variables\n'.format(len(categorical)))
#
# print('The categorical variables are :', categorical)

dep_var = 'Purchased'

X = df.drop([dep_var], axis=1)
y = df[dep_var]

# try:
#     print(check_classification_targets(y, True))
# except:
#     print("continous")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
# print(X_train)
# print(y_train)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
print(X_test)
print(X_test.shape)
X_test = sc_X.transform(X_test)
print(X_test)
print(X_test.shape)

automl = AutoML()
model = automl.fit(X_train, y_train, task="classification", time_budget=60, max_iter=50)
# y_pred = [np.argmax(i) for i in automl.predict_proba(X_test)]
# print(y_pred)
# Export the best model

print('Best Learner Found:', automl.best_estimator)
print('Ideal hyperparmeter config:', automl.best_config)
print('Highest accuracy on validation data: {0:.4g}'.format(1 - automl.best_loss))
print('Training duration of run: {0:.4g} s'.format(automl.best_config_train_time))
print(automl.model.estimator)
y_pred = automl.predict(X_test)
y_pred_proba = automl.predict_proba(X_test)[:,1]

print('accuracy found ', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred, y_test))
print('roc_auc found', '=', 1 - sklearn_metric_loss_score('roc_auc', y_pred_proba, y_test))
print('log_loss found', '=', sklearn_metric_loss_score('log_loss', y_pred_proba, y_test))

accuracy = '%.2f' % (metrics.accuracy_score(y_test, y_pred) * 100) + ' %'
recall = '%.2f' % (metrics.recall_score(y_test, y_pred) * 100) + ' %'
f1 = '%.2f' % (metrics.f1_score(y_test, y_pred) * 100) + ' %'
precision = '%.2f' % (metrics.precision_score(y_test, y_pred) * 100) + ' %'
# print("accuracy:", accuracy)
# print("recall:", recall)
# print("f1:", f1)
# print("precision:", precision)
# print(automl.model)
# print(automl.model)
# print(automl.model)
# print(automl.model)
sys.exit()
# regressor = LogisticRegression()
#
# regressor.fit(X_train, y_train)
#
# y_pred = regressor.predict(X_test)
# # print(regressor.score(X_test, y_test))
#
# r2 = regressor.score(X_test, y_test)
#
# # sns.histplot(y_test - y_pred, )
# # plt.show()
#
# cm = confusion_matrix(y_test, y_pred)
# # for i in cm:
# # print("TN", cm[0][0], "FP", cm[0][1])
# # print("FN", cm[1][0], "TP", cm[1][1])
# from sklearn.metrics import classification_report
#
# class_report = classification_report(y_test, y_pred, output_dict=True)
#
# # for i in class_report:
# print(class_report)

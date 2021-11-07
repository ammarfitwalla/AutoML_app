import warnings
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import sys
from sklearn.multiclass import check_classification_targets
import matplotlib.pyplot as plt
import seaborn as sns

file_name = r'D:\KS\Projects\django_ml\media\documents\Salary_Data.csv'
# file_name = r'D:\AF\Projects\Tensorflow_Certification\Machine_Learning\Linear_Regression\Multiple_LR\1_multiple_salary.csv'
# print(os.extsep)

df = pd.read_csv(file_name)
# print(df.head())

# sns.pairplot(df)
# plt.show()

# categorical = [var for var in df.columns if df[var].dtype == 'O']
#
# print('There are {} categorical variables\n'.format(len(categorical)))
#
# print('The categorical variables are :', categorical)

dep_var = 'Salary'
# dep_var = 'Gender'

X = df.drop([dep_var], axis=1)
y = df[dep_var]

# print(df['Salary'].nunique())

# sys.exit()
try:
    print(check_classification_targets(y, True))
except:
    print("continous")

# if y.dtype == 'O':
#     print('categorical')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
# print(X_train)
# print(y_train)

regressor = LinearRegression(fit_intercept=True, normalize=True, n_jobs=10, positive=True)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(regressor.score(X_test, y_test))

r2 = regressor.score(X_test, y_test)

# sns.distplot(y_test - y_pred, )
# plt.show()

# coeffecients = pd.DataFrame(regressor.coef_, X.columns)
# coeffecients.columns = ['Coeffecient']
# print(coeffecients)
#
# g = plt.scatter(y_test, y_pred)
# g.axes.set_yscale('log')
# g.axes.set_xscale('log')
# g.axes.set_xlabel('True Values ')
# g.axes.set_ylabel('Predictions ')
# g.axes.axis('equal')
# g.axes.axis('square')

plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()

# print(y_pred)
#
# print(metrics.mean_absolute_error(y_test, y_pred))
# print(metrics.mean_squared_error(y_test, y_pred))
# print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# plt.style.use('default')
# plt.style.use('ggplot')
#
# fig, ax = plt.subplots(figsize=(8, 4))
#

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="green")
# plt.title("Salary vs Experience (Training set)")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
plt.show()
#
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="green")
# plt.title("Salary vs Experience (Testing set)")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
plt.show()
#
# plt.style.use('default')
# plt.style.use('ggplot')

# fig, ax = plt.subplots(figsize=(7, 3.5))
#
# ax.plot(X_test, y_pred, color='k', label='Regression model')
# ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
# ax.set_ylabel('Gas production (Mcf/day)', fontsize=14)
# ax.set_xlabel('Porosity (%)', fontsize=14)
# ax.legend(facecolor='white', fontsize=11)
# ax.text(0.55, 0.15, '$y = %.2f x_1 - %.2f $' % (regressor.coef_[0], abs(regressor.intercept_)), fontsize=17,
#         transform=ax.transAxes)
#
# fig.tight_layout()

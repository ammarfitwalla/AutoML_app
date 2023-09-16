# import pandas as pd
# import numpy as np
# from flaml import AutoML
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import max_error, mean_absolute_error, mean_squared_log_error, mean_squared_error, r2_score
#
# dataset = load_boston()
#
# x, y = dataset.data, dataset.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
#
# automl_reg = AutoML()
# automl_reg.fit(x_train, y_train, task="regression")
#
# y_pred = automl_reg.predict(x_test)
#
# print('max error value :', max_error(y_test, y_pred))
# print('mean absolute error value :', mean_absolute_error(y_test, y_pred))
# print('mean squared error :', mean_squared_error(y_test, y_pred))
# print("mean squared log error :", mean_squared_log_error(y_test, y_pred))
# print("r2 score :", r2_score(y_test, y_pred))


# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Sample data (replace with your dataset)
# data = pd.DataFrame({
#     'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female'],
#     'Buy': [1, 1, 0, 0, 1, 0, 1, 0]
# })
#
# # Create a countplot to visualize the counts
# plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
# sns.countplot(data=data, x='Gender', hue='Buy')
# plt.xlabel('Gender')
# plt.ylabel('Count of Males and Females')
# plt.legend(title='Buy', labels=['Not Bought', 'Bought'])
#
# plt.show(block=True)



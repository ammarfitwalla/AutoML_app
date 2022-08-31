# example of a ordinal encoding
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import category_encoders as ce

df = pd.read_csv('employees.csv')
print(df.columns)
# genderdf = df['Gender']

# print(genderdf.head())

# ord_encoder = ce.OrdinalEncoder(cols=['Gender'])
# ord_result = ord_encoder.fit_transform(genderdf)
# print(ord_result)
# print('----------------------------------------------------------------')

df = pd.get_dummies(df[['Gender', 'Senior Management', 'Team']], prefix="", prefix_sep="")
# df = oh_encoder.fit_transform(df)
print(df)

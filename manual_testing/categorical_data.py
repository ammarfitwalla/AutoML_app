# example of a ordinal encoding
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import category_encoders as ce

df = pd.read_csv('employees.csv')

genderdf = df['Gender']

# print(genderdf.head())

ord_encoder = ce.OrdinalEncoder(cols=['Gender'])
oh_encoder = ce.OneHotEncoder(cols=['Gender'])

ord_result = ord_encoder.fit_transform(genderdf)
oh_result = oh_encoder.fit_transform(genderdf)

print(ord_result)
print('----------------------------------------------------------------')
print(oh_result)
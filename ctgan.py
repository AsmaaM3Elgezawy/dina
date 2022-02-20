import pandas as pd

data = pd.read_csv('heart.csv')
data.drop(['age'], axis=1, inplace=True)
print(data.columns)
categorical_features = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
       'oldpeak', 'slope', 'ca', 'thal', 'target']


import ctgan 
from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer(verbose=True)
ctgan.fit(data, categorical_features, epochs = 300)

samples = ctgan.sample(1000)

print(samples.head())
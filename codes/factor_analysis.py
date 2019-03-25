import numpy as np
import pandas as pd
import csv
import sklearn.decomposition as sk_d

## data input
train_data = pd.read_csv('train.csv',header=0)
train_features = train_data.iloc[0:,2:]
train_label = train_data.iloc[0:,1]

test_data = pd.read_csv('test.csv',header=0)
test_features = test_data.iloc[0:,2:]
test_label = test_data.iloc[0:,1]

## Factor Analysis
def fa(train, test, n_component):
    transformer = sk_d.FactorAnalysis(n_components=n_component, random_state=0)
    train_out = transformer.fit_transform(train)
    test_out = transformer.fit_transform(test)
    return train_out, test_out

n_component = 100
train_trans,test_trans = fa(train_features,test_features,n_component)

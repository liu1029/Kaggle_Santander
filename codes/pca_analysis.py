# Created by Jie Liu March 18th 2019


import numpy as np
import pandas as pd
import sklearn.decomposition as sk_d
import matplotlib.pyplot as plt

import sklearn.feature_selection as fs
from sklearn.externals import joblib
from steppy.base import BaseTransformer
import os

# function define
def mypca(data_train,data_test,n_components):
    # Dimension reduction PCA:
    # input: train_data
    # ourput: train_data_pca
    pca = sk_d.PCA(n_components=n_components)
    pca.fit(data_train)

    data_train_out = pca.transform(data_train)
    data_test_out = pca.transform(data_test)

    return data_train_out,data_test_out,pca

# read data
# first column : label  first row: head
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_features = train_data.iloc[0:,2:]
train_label= train_data.iloc[0:,1]

test_features = test_data.iloc[0:,1:]

'''
# scree plot
train_features_pca, test_features_pca, pca  = mypca(train_features,test_features,500)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()

'''
train_features_pca, test_features_pca, pca  = mypca(train_features,test_features,100)

'''
# Add column names to train_data_pca
column_names = map(lambda a,b: str(a)+str(b), ["componet"]*50, range(1, 50))
# converting map object to set
numbersSquare = set(column_names)

'''

# # Dimension reduction PCA (50 coponents):
# svd = sk_d.TruncatedSVD(n_components=50, n_iter=7, random_state=42)
# svd.fit(train_data.iloc[1:,1:])


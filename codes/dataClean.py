# Created by Jie Liu March 18th 2019
import numpy as np
import pandas as pd
import sklearn.decomposition as sk_d
import sklearn.feature_selection as fs
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from sklearn.decomposition import PCA
import os
from sklearn.decomposition import TruncatedSVD

# read data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Dimension reduction PCA input
pca = sk_d.PCA(n_components=50)
pca.fit(train_data.iloc[1:,1:])
train_data_pca = pca.transform(train_data.iloc[1:,1:])


column_names = map(lambda a,b: str(a)+str(b), ["componet"]*50, range(1, 50) )
# converting map object to set
numbersSquare = set(column_names)
train_data_pca.columns = ["Sequence", "Start", "End", "Coverage"]



#
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
svd.fit(train_data.iloc[1:,1:])
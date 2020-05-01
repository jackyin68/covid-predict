from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures

import os

cwd = os.path.dirname(os.getcwd())
print(cwd)
# Get Data Frame from local parquet file
df = pd.read_parquet(r'path')

# Slice target feature
y = df["target"]
# Slice input feature and split target from input features.
X = df.drop(columns=["target"])

poly = PolynomialFeatures(degree=6).fit(X)
X = poly.transform(X)
print("Get Polynomial Features: ", poly.get_feature_names())

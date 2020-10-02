import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

#%%
def plot_variable_pairs(df):
    return sns.pairplot(df, kind='reg')

# %%
def plot_categorical_and_continuous_vars(df, categorical, continuous):
    plt.subplot(221)
    box = sns.boxplot(data=df, x=categorical, y=continuous)
    plt.subplot(222)
    swarm = sns.swarmplot(data=df, x=categorical, y=continuous)
    plt.subplot(223)
    violin = sns.violinplot(data=df, x=categorical, y=continuous)

# %%
def corr_heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, cmap='Blues', annot=True)

# %%
def select_kbest(X, y, n):
    """
    Returns the top n selected features based on the SelectKBest calss
    Parameters: predictors(X) in df, target(y) in df, the number of features to select(n)
    """
    f_selector = SelectKBest(f_regression, k=n)
    f_selector = f_selector.fit(X, y)
    f_support = f_selector.get_support()
    f_feature = X.iloc[:, f_support].columns.tolist()
    return f_feature

# %%
def rfe(X, y, n):
    """
    Returns the top n selected features based on the RFE calss
    Parameters: predictors(X) in df, target(y) in df, the number of features to select(n)
    """
    lm = LinearRegression()
    rfe = RFE(lm, n)
    rfe = rfe.fit(X, y)
    rfe_support = rfe.support_
    f_feature = X.iloc[:, rfe_support].columns.tolist()
    return f_feature
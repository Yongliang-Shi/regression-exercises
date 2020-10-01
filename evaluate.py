import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
# %%
def evaluate_slr(target, feature, df):
    """
    Returns evaluation table for simple linear regression model
    Parameters: target var(str), feature(str), data
    """
    baseline = df[target].mean()
    formula = target + ' ~ ' + feature
    model = ols(formula, df).fit()
    evaluate = df[[feature, target]]
    evaluate['baseline'] = baseline
    evaluate[(target + '_pred')] = model.predict()
    evaluate['baseline_residual'] = evaluate[target] - evaluate.baseline
    evaluate['model_residual'] = evaluate[target] - evaluate[(target + '_pred')]
    return evaluate

# %%
def plot_residuals(actual, predicted):
    """
    Returns the scatterplot of actural y in horizontal axis and residuals in vertical axis
    Parameters: actural y, predicted y
    """
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()
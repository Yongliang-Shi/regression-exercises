import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%
def evaluate_slr(target, feature, df):
    """
    Returns evaluation table for simple linear regression model
    Parameters: target var(str), feature(str), data
    Baseline: mean
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
    Parameters: actural y(df.se), predicted y(df.se)
    """
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()
# %%
def slr_metrics(target, feature, df):
    """
    Returns a list of SSE, MSE, RMSE, $R^2$, and f_pval
    Parameters: target var(str), feature(str), data source(df)
    Baseline: mean
    """
    baseline = df[target].mean()
    formula = target + ' ~ ' + feature
    model = ols(formula, df).fit()
    x = df[feature]
    y_actural = df[target]
    y_pred = model.predict()
    baseline_residual = y_actural - baseline
    baseline_sse = (baseline_residual**2).sum()
    model_residual = y_actural - y_pred
    model_sse = (model_residual**2).sum()
    model_mse = mean_squared_error(y_actural, y_pred)
    model_rmse = sqrt(model_mse)
    r2 = model.rsquared
    f_pval = model.f_pvalue
    metrics = [baseline_sse, model_sse, model_mse, model_rmse, r2, f_pval]
    return metrics
# %%
def compare_slr_metrics(target, features, df):
    """
    Compare the metrics of simple linear regression on multiple continuous features
    Parameters: target var(str), interested features(list), data source(df)
    Prerequisite: import function `slr_metrics`
    Baseline: mean
    """
    metrics = pd.DataFrame(index=['baseline_sse', 'model_sse', 'model_mse', 'model_rmse', 'r2', 'f_pval'])
    for feature in features:
        col = slr_metrics(target, feature, df)
        metrics[feature] = col
    return metrics.T
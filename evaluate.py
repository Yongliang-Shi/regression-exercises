# %%
def evaluate_slr(target, feature, df):
    '''
    Returns evaluation table for simple linear regression model
    Parameters: target var(str), feature(str), data
    '''
    baseline = df[target].mean()
    formula = target + ' ~ ' + feature
    model = ols(formula, df).fit()
    evaluate = df[[feature, target]]
    evaluate['baseline'] = baseline
    evaluate[(target + '_pred')] = model.predict()
    evaluate['baseline_residual'] = evaluate[target] - evaluate.baseline
    evaluate['model_residual'] = evaluate[target] - evaluate[(target + '_pred')]
    return evaluate
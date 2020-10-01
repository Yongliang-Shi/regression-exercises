import matplotlib.pyplot as plt
import seaborn as sns
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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from pydataset import data

import warnings
warnings.filterwarnings("ignore")

def cleaning(titanic):
    drop_index = titanic[titanic.embarked.isnull()].index
    titanic.drop(index=drop_index, inplace=True)
    titanic.drop(columns=['deck', 'passenger_id'], inplace=True)
    dummies = pd.get_dummies(titanic[['sex','embarked']], drop_first=True)
    titanic = pd.concat([titanic, dummies], axis=1)
    titanic.drop(columns=['sex', 'embarked','class','embark_town'], inplace=True)
    return titanic

def cleaning_spliting(titanic):
    titanic = cleaning(titanic)
    train_validate, test = train_test_split(titanic, test_size=0.2, 
                                            random_state=123,
                                            stratify=titanic.survived
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived
                                      )
    return train, validate, test

def prep_titanic(titanic):
    train, validate, test = cleaning_spliting(titanic)
    imputer = SimpleImputer(strategy = 'most_frequent')
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test

def prep_titanic_mean(titanic):
    train, validate, test = cleaning_spliting(titanic)
    imputer = SimpleImputer(strategy = 'mean')
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test

def prep_iris(iris):
    iris.drop(columns=['species_id','measurement_id'], inplace=True)
    iris.rename(columns={'species_name':'species'}, inplace=True)
    species_dummy = pd.get_dummies(iris[['species']])
    iris = pd.concat([iris, species_dummy], axis=1)
    train_validate, test = train_test_split(iris, test_size=.2, random_state=123, stratify=iris.species)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.species)
    return train, validate, test

def prep_mall(df):
    '''
    Takes the acquired mall data, does data prep, and returns
    train, test, and validate data splits.
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_validate, test = train_test_split(df, test_size=.15, random_state=442)
    train, validate = train_test_split(train_validate, test_size=.15, random_state=442)
    return train, validate, test
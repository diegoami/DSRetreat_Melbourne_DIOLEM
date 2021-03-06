from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import sys

sys.path.append('..')
import pandas as pd


def extract(df_train, df_test):
    # print(df_train.columns)
    # print(df_test.columns)
    # df_train = df_train.drop('date', axis=1)
    # df_test = df_test.drop('date', axis=1)
    # print(df_train.head())
    # print(df_test.head())

    # df_train.fillna(0, inplace=True)
    # df_test.fillna(0, inplace=True)

    # train_columns = [x for x in df_train.columns if x not in ['id', 'granted']]
    #
    # test_columns = [x for x in df_train.columns if x not in ['id', 'granted']]

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # print(X_train.iloc[:5,:])
    # print(y_train.iloc[:5])
    # print(X_test.iloc[:5, :])
    # print(y_test.iloc[:5])

    return X_train, y_train, X_test, y_test






def extract_old(df_train, df_test):
    cat_columns = ['sponsor', 'grant_category', 'Dept.No.', 'Faculty.No.']
    for cat_column in cat_columns:
        df_train[cat_column] = df_train[cat_column].astype('category')
        df_train[cat_column] = df_train[cat_column].cat.codes
        df_test[cat_column] = df_test[cat_column].astype('category')
        df_test[cat_column] = df_test[cat_column].cat.codes

    #    df_train = df_train.drop('date', axis=1)
#    df_test = df_test.drop('date', axis=1)

    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)
    print(df_train.columns)
    print(df_test.columns)

    relevant_columns = [x for x in set(list(df_train.columns)+list(df_test.columns)) if x not in ['id', 'granted', 'Unnamed: 0']]
    #print(relevant_columns)
    missing_train_columns = [x for x in relevant_columns if x not in df_train ]


    missing_test_columns = [x for x in relevant_columns if x not in df_test ]
  #  print(missing_test_columns)

    df_train.reindex(columns=list(df_train.columns)+missing_train_columns)

    df_test = df_test.reindex(columns=(list(df_test.columns) + missing_test_columns))

    X_train = df_train[relevant_columns]
    y_train = df_train['granted']
    X_test = df_test[relevant_columns]
    y_test = df_test['granted']

    y_train.fillna(0, inplace=True)
    y_test.fillna(0, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    return X_train, y_train, X_test, y_test



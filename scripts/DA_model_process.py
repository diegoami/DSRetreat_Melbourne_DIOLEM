from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import sys

sys.path.append('..')
import pandas as pd


def extract(df):
    cat_columns = ['sponsor', 'grant_category']
    for cat_column in cat_columns:
        df[cat_column] = df[cat_column].astype('category')
        df[cat_column] = df[cat_column].cat.codes
    df = df.drop('date', axis=1)
    relevant_columns = [x for x in df.columns if x not in ['id', 'granted']]
    X = df[relevant_columns]
    y = df['granted']
    return X, y


if __name__ == "__main__":

    df_train = pd.read_csv('../data/train_apps_rfcd_seo_mod.csv', parse_dates=True)
    df_test = pd.read_csv('../data/test_apps_rfcd_seo_mod.csv', parse_dates=True)

    X_train, y_train = extract(df_train)
    X_test, y_test= extract(df_test)
    print(X_train)
    print(y_train)

    classifier = RandomForestClassifier( n_estimators=2000)
    classifier.fit(X=X_train, y=y_train)

    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))

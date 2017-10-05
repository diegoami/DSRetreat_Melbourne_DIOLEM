from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
import numpy as np

import sys

sys.path.append('..')
from scripts.kaggle_solver import KaggleSolver
import pandas as pd

if __name__ == "__main__":

    df_train = pd.read_csv('../data/train_apps_rfcd_seo_mod.csv', parse_dates=True)

    cat_columns = ['sponsor', 'grant_category', 'sponsor_c', 'grant_category_c']
    for cat_column in cat_columns:
        df_train[cat_column] = df_train[cat_column].astype('category')
        df_train[cat_column] = df_train[cat_column].cat.codes
    df_train = df_train.drop('date', axis=1)

    relevant_columns = [x for x in df_train.columns if x not in ['id', 'granted']]

    X_train = df_train[relevant_columns]
    y_train = df_train['granted']

    classifier = RandomForestRegressor()
    classifier.fit(X=X_train, y=y_train)




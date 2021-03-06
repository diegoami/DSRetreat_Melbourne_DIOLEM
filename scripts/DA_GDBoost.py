from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from scripts.DA_model_extract import extract, extract_old
import sys
from itertools import product
from scripts.eo_transport_data import run
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn import linear_model, decomposition, datasets
print(__doc__)
def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    return X_train, y_train, X_test, y_test




param_grid1 = {
  # 'n_estimators': [200, 400, 800, 1500],
    'min_samples_split' : [2,3,4,5,6],
    'min_samples_leaf' : [0.5,0.4,0.3,0.2,0.1,0.05],
                 'max_depth' : [3,4,5,6],
}


param_grid2 = {
  # 'n_estimators': [200, 400, 800, 1500],
    'min_samples_split' : [2,3],
    'min_samples_leaf' : [0.15,0.12, 0.1,0.08, 0.05],
                 'max_depth' : [4,5,6],
        'max_features' : [5,7,11,15,20],
}

param_grid3 = {
  # 'n_estimators': [200, 400, 800, 1500],
    'min_samples_split' : [2,3,4],
    'min_samples_leaf' : [0.06,0.08, 0.07],
                 'max_depth' : [6,7,8],
        'max_features' : [18,20,22,24],
    'n_estimators' : [100,500,1000,3000]
}

n_estimators=1
def main2(**kwargs):


    best_roc, best_args = 0, None
    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    for args in iter_param_grid(param_grid3):  # tried 5,6,7,8,9

        classifier = make_pipeline(
            MinMaxScaler(),
            GradientBoostingRegressor(**args).fit(X_train, y_train)

        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        rs = roc_auc_score(y_test, y_pred)
        print(rs, args)
        if (rs > best_roc):
            best_roc, best_args = rs, args
    print("=== BEST PARAMETERS ===")
    print(best_roc, best_args)
    return classifier

def iter_param_grid(p):

    # Always sort the keys of a dictionary, for reproducibility
    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params






        #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":

    main2(print_info=False)
    classifier = main2(print_info=False)
    df_train_all, df_hold = run(ds1='train_all', ds2='hold')
    X_train, y_train, X_test, y_test = extract(df_train_all, df_hold )
    y_pred = classifier.predict(X_test)
    rs = roc_auc_score(y_test, y_pred)
    print(rs)
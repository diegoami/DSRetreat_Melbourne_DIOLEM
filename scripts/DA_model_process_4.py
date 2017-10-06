from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from scripts.DA_model_extract import extract, extract_old
import sys
from scripts.eo_transport_data import run
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from itertools import product
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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline


def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    return X_train, y_train, X_test, y_test

xgbparams1 = {
             'min_child_weight' : [2,4, 6,8, 10],
             'max_depth' : [2,3,4, 5,6,7],
             #'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0,7, 0.8],
             #'subsample': [0.5,  0.6, 0.7, 0.8, 0.9, 1],
             #'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
             #'reg_alpha': [0.1, 0.01 , 1, 10, 100]

        }


xgbparams2 = {
             'min_child_weight' : [3,4, 5],
             'max_depth' : [2, 3, 4],
             'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0,7, 0.8],
             #'subsample': [0.5,  0.6, 0.7, 0.8, 0.9, 1],
             #'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
             #'reg_alpha': [0.1, 0.01 , 1, 10, 100]

        }

xgbparams3 = {
             'min_child_weight' : [3,4, 5],
             'max_depth' : [2, 3, 4],
             'gamma': [0.1, 0.2, 0.3],
             'subsample': [0.5,  0.6, 0.7, 0.8, 0.9, 1],
             'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
             #'reg_alpha': [0.1, 0.01 , 1, 10, 100]

        }


xgbparams4 = {
             'min_child_weight' : [3],
             'max_depth' : [3],
             'gamma': [0.1, 0.2, 0.3],
             'subsample': [0.5,  0.6, 0.7],
             'colsample_bytree': [0.7, 0.8, 0.9],
             'reg_alpha': [0.1, 0.01 , 1, 10, 100]

        }

xgbparams5 = {
             'min_child_weight' : [3],
             'max_depth' : [3],
             'gamma': [0.3, 0.4],
             'subsample': [0.6],
             'colsample_bytree': [ 0.8],
             'reg_alpha': [0.01 ],
            'learning_rate' : [0.1, 0.01, 0.001]

        }

xgbparams6 = {
             'min_child_weight' : [3],
             'max_depth' : [3],
             'gamma': [ 0.4, 0.5],
             'subsample': [0.6],
             'colsample_bytree': [ 0.8],
             'reg_alpha': [0.01 ],
            'learning_rate' : [ 0.01],
            'n_estimators' : [100,500,1000,5000]


        }

xgbparams7 = {
             'min_child_weight' : [2,3,4],
             'max_depth' : [2,3,4],
             'gamma': [ 0.4, 0.5, 0.6],
             'subsample': [0.5,0.6,0.7],
             'colsample_bytree': [ 0.7,0.8,0.9],
             'reg_alpha': [0.01 ],
            'learning_rate' : [ 0.01,0.001],
            'n_estimators' : [1000,2000,3000]


        }

def main(**kwargs):
    best_roc, best_args = 0, None
    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    for args in iter_param_grid(xgbparams7):  # tried 5,6,7,8,9

        classifier = make_pipeline(
           MinMaxScaler(),
          # PCA(),
            XGBRegressor(**args)
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

   # main_old(print_info=False)
   #main(print_info=False)
   classifier = main(print_info = False)
   df_train_all, df_hold = run(ds1='train_all', ds2='hold')
   X_train, y_train, X_test, y_test = extract(df_train_all, df_hold )
   y_pred = classifier.predict(X_test)
   rs = roc_auc_score(y_test, y_pred)
   print(rs)
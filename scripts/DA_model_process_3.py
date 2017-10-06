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
from sklearn.linear_model import Lasso


def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    return X_train, y_train, X_test, y_test







def main(**kwargs):
    best_roc, best_params= 0, None
    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test  = extract(df_train, df_test)
    for min_child_weight in  [7]: # tried 5,6,7,8,9
        for max_depth in [6]: # tried 5,6,7,8
            for gamma in [0.4]: # tried 0,0.1,0.2,0.3,0.4,0.5
                for subsample in [0.7]: # tried 0.6, 0.7, 0.8, 0.9
                    for colsample_bytree in [  0.9]: # tired 0.5, 0.7, 0.9
                        for reg_alpha in [0.1, 0.01 , 1, 10, 100]:  #

                            classifier = XGBRegressor(max_depth=max_depth, min_child_weight=min_child_weight,gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha)
                            classifier.fit(X_train, y_train)
                            y_pred = classifier.predict(X_test)
                            rs = roc_auc_score(y_test, y_pred )
                            print(rs, min_child_weight, max_depth, gamma, subsample, colsample_bytree,reg_alpha)
                            if (rs > best_roc):
                                best_roc, best_params = rs, (min_child_weight, max_depth, gamma, subsample, colsample_bytree, reg_alpha)
    print(best_roc, best_params, gamma, subsample, colsample_bytree,reg_alpha )




    #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":

    main(print_info=False)
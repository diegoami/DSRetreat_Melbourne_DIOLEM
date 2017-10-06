import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from scripts.DA_model_extract import extract, extract_old
import sys
from scripts.eo_transport_data import run
import pandas as pd
import numpy as np

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


from sklearn.decomposition import PCA

def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)

    return X_train, y_train, X_test, y_test




def do_ntw(X_train,y_train, X_test, y_test):
    classifier1 = make_pipeline(
        MinMaxScaler(),
        XGBRegressor(**{'colsample_bytree': 0.8, 'gamma': 0.1, 'max_depth': 3, 'min_child_weight': 3, 'reg_alpha': 0.01, 'subsample': 0.6})
    )

    classifier2 = make_pipeline(
        MinMaxScaler(),
        PCA(),
        RandomForestRegressor(
            **{'max_features': 8, 'min_samples_leaf': 26, 'min_samples_split': 4, 'n_estimators': 600, 'n_jobs': -1})
    )
    classifiers = [classifier1, classifier2]

    model = Sequential()
    model.add(Dense(500, input_dim=2, activation='relu'))
    #   model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    #   model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'],
                  )

    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    y_out_test_list = [classifier.predict(X_test) for classifier in classifiers]
    y_out_test = np.vstack(y_out_test_list)

    y_out_train_list = [classifier.predict(X_train) for classifier in classifiers]
    y_out_train = np.vstack(y_out_train_list)



    model.fit(y_out_train.T , y_train,
              epochs=200,
              batch_size=256,validation_data=(y_out_test.T,y_test)
              )
    #score = model.evaluate(X_test, y_test)
    y_pred = model.predict_proba(y_out_test.T )
    return y_pred, classifiers, model


    #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":
    df_train, df_test = run(print_info=False)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    y_pred, classifiers, model = do_ntw(X_train, y_train, X_test, y_test)
    print(roc_auc_score(y_test,y_pred))
    df_train_all, df_hold = run(ds1="train_all", ds2="hold", print_info=False)
    X_train_all, y_train_all, X_test_hold, y_test_hold = extract(df_train_all, df_hold )

    y_out_hold_list = [classifier.predict(X_test_hold) for classifier in classifiers]
    y_out_hold = np.vstack(y_out_hold_list)
    y_pd_hold = model.predict_proba(y_out_hold.T)
    print(roc_auc_score(y_test_hold, y_pd_hold))



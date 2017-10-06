import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from scripts.DA_model_extract import extract, extract_old
import sys
from scripts.eo_transport_data import run
import pandas as pd
import numpy as np



def extract(df_train, df_test):


    df_main_train = pd.read_csv('../data/' + 'train' + '.csv', low_memory=False, parse_dates=['date']).loc[:,['id','RFCD.Code.1','SEO.Code.1']]
    df_main_test = pd.read_csv('../data/' + 'test' + '.csv', low_memory=False, parse_dates=['date']).loc[:,['id','RFCD.Code.1','SEO.Code.1']]

    df_main_train = df_main_train.fillna(0)
    df_main_test = df_main_test.fillna(0)


    df_train = df_train.merge(df_main_train,how = 'left', on='id')
    df_test = df_test.merge(df_main_test,how = 'left', on='id')

    y_train = df_train['granted']
    y_test = df_test['granted']

    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)




    return X_train, y_train, X_test, y_test


def print_best_parameters(classifier):
    if hasattr(classifier, "best_estimator_"):
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


def test_with_classif(classifier, df_train, df_test):

    X_train, y_train, X_test, y_test  = extract(df_train, df_test)
    classifier.fit(X=X_train, y=y_train)
    y_pred = classifier.predict(X_test)
    print_best_parameters(classifier)
    print(roc_auc_score(y_test, y_pred))
    if (hasattr(classifier, "_coef")):
        coef = pd.Series(classifier.coef_, index=X_train.columns)
        print(coef)

def test_with_classifs(classifiers, df_train, df_test):


    X_train, y_train, X_test, y_test  = extract(df_train, df_test)
    y_preds = [classifier.predict(X_test) for classifier in classifiers]
    y_pred_concat = np.concatenate(y_preds)
    y_pred_average = y_pred_concat.mean()
    print(roc_auc_score(y_test, y_pred_average))



def main(**kwargs):

    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)

    keep = ['A', 'A.', 'B', 'C', 'Dept.No.', 'Dept.No._s_rate',
       'EXTERNAL_ADVISOR', 'EXT_CHIEF_INVESTIGATOR', 'Faculty.No.',
       'Faculty.No._s_count', 'Faculty.No._s_rate',
       'Number.of.Successful.Grant', 'Number.of.Unsuccessful.Grant',
       'STUDRES', 'STUD_CHIEF_INVESTIGATOR', 'With.PHD', 'date',
       'grant_category', 'grant_category_s_count', 'grant_category_s_rate',
       'grant_value', 'max_year_of_birth', 'max_years_in_uni',
       'mean_year_of_birth', 'mean_years_in_uni', 'month', 'n_internals',
       'rfcd_l0_s_count', 'rfcd_l0_s_rate', 'rfcd_l1_s_count',
       'rfcd_l1_s_rate', 'rfcd_l2_s_count', 'rfcd_l2_s_rate', 'seo_l0_s_count',
       'seo_l0_s_rate', 'seo_l1_s_count', 'seo_l1_s_rate', 'seo_l2_s_count',
       'seo_l2_s_rate', 'sponsor', 'sponsor_s_count', 'sponsor_s_rate',
       'RFCD.Code.1', 'SEO.Code.1']

    X_train = X_train.loc[:, X_train.columns.isin(keep)]
    X_test = X_test.loc[:, X_test.columns.isin(keep)]


    print(X_train.info())

    print(X_train.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(100,  input_dim=len(keep), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(X_train, y_train,
              epochs=100,
              batch_size=64,
              shuffle=True)

    y_pred = model.predict_proba(X_test)

    print('\n roc_auc_score: ')
    print(roc_auc_score(y_test, y_pred))






def main2(**kwargs):

    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    print(X_train.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    look_back = 1

    model = Sequential()
    model.add(Dense(64,  input_dim=88, activation='relu'))
    model.add(LSTM(4, input_shape=(88, look_back)))

    model.add(Dense(500, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(X_train, y_train,
              epochs=100,
              batch_size=32)

    y_pred = model.predict_proba(X_test)

    print('\n roc_auc_score: ')
    print(roc_auc_score(y_test, y_pred))

    #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":

    main(print_info=False)
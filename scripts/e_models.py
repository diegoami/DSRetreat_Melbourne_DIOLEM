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



def extract(df_train, df_test):

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
    print(X_train.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64,  input_dim=88, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(X_train, y_train,
              epochs=200,
              batch_size=256)
    score = model.evaluate(X_test, y_test)
    print(score)


    #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":

    main(print_info=False)

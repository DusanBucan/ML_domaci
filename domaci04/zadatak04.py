import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def label_encoding(data, name):
    data[name] = data[name].astype('category').cat.codes
    return data


def categorical_data(data):
    data = label_encoding(data, 'dead')
    data = label_encoding(data, 'sex')
    data = label_encoding(data, 'airbag')
    data = label_encoding(data, 'seatbelt')
    data = label_encoding(data, 'abcat')
    data = label_encoding(data, 'occRole')
    
    return data


def preprocess_data(data):
    data = categorical_data(data)

    return data


def calculate_micro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Uklanjanje NaN vrednosti iz trening skupa
    train_data = train_data.dropna()

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    Y_train = train_data['speed'].to_numpy()
    del train_data['speed']
    X_train = train_data.to_numpy()

    from sklearn.ensemble import BaggingClassifier
    # clf = BaggingClassifier(n_estimators=20, max_samples=0.5, max_features=0.5)

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=1300, max_features='auto', learning_rate=0.1)

    # from sklearn.svm import LinearSVC
    # from sklearn.feature_selection import SelectFromModel
    # lsvc = LinearSVC(C=2, penalty="l1", dual=False).fit(X_train, Y_train)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_train = model.transform(X_train)
    # print(X_train.shape)


    # clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, Y_train)

    Y_test = test_data['speed'].to_numpy()
    del test_data['speed']
    X_test = test_data.to_numpy()

    Y_predict = clf.predict(X_test)
    score = calculate_micro_f1_score(Y_test, Y_predict)
    print(score)

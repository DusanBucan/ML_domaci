import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


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


def statistic(tr_data):
    cnt_per_class = {}
    for tr_instance in tr_data.values:
        speed_class = tr_instance[0]
        if speed_class in cnt_per_class.keys():
            cnt_per_class[speed_class] += 1
        else:
            cnt_per_class[speed_class] = 1

    objects = cnt_per_class.keys()
    y_pos = np.arange(len(objects))
    performance = [cnt_per_class[ob] for ob in objects]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.show()


def crossValidation(X, Y):

    # params = {"n_estimators": [300, 700, 900], "learning_rate": [0.1, 0.2, 0.3]}
    # grid = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=params, cv=10)
    # params = {"C": [1, 1.4, 1.6, 2, 2.5]}
    # grid = GridSearchCV(estimator=SVC(), param_grid=params, cv=10)
    grid = grid.fit(X, Y)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)

    # clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
    # scores = cross_val_score(clf, X, Y, cv=5)
    # return  scores.mean()


def scale_data(X, scaler=None):
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(X)
    new_X = scaler.transform(X)
    return new_X, scaler


def featureSelectionTreBased(X, y=None, selector=None):
    if selector == None:
        selector = ExtraTreesClassifier(n_estimators=50)
        selector = selector.fit(X, y)

    model = SelectFromModel(selector, prefit=True)
    X_new = model.transform(X)
    return X_new, selector


# sto manje C to ce MANJE obelezja da ostane
def featureSelectorSVMBased(X, y=None, selector=None):
    if selector == None:
        selector = LinearSVC(C=0.01, penalty="l1", dual=False)
        selector = selector.fit(X, y)

    model = SelectFromModel(selector, prefit=True)
    X_new = model.transform(X)
    return X_new, selector


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Uklanjanje NaN vrednosti iz trening skupa
    train_data = train_data.dropna()
    print(len(train_data))

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    statistic(train_data)


    Y_train = train_data['speed'].to_numpy()
    del train_data['speed']
    X_train = train_data.to_numpy()

    # sklairana koriscenjem Normalne Raspodele
    X_train, scaler = scale_data(X_train)

    #izdvajanje obelezja uz pomoc Stabla
    # X_train, selector = featureSelectionTreBased(X_train, Y_train, None)

    # crossValidation(X_train, Y_train)
    #
    # clf = SVC(C=2)
    clf = AdaBoostClassifier(learning_rate=0.1, n_estimators=900)
    # clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, Y_train)
    #
    Y_test = test_data['speed'].to_numpy()
    del test_data['speed']
    X_test = test_data.to_numpy()

    X_test, scaler = scale_data(X_test, scaler)
    # X_test, selector = featureSelectionTreBased(X_test, None, selector)

    Y_predict = clf.predict(X_test)
    score = calculate_micro_f1_score(Y_test, Y_predict)
    print(score)

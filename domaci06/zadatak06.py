import sys
# import random
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import shapiro
# from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from scipy.stats import shapiro


def label_encoding(data, name, le=None):
    if le is None:
        le = LabelEncoder()
        le = le.fit(list(set(data[name])))
    data[name] = le.transform(data[name])
    return le


def one_hot_encoding(data, name, lb=None):
    if lb is None:
        lb = LabelBinarizer()
        data = data.join(pd.DataFrame(lb.fit_transform(data[name]), columns=lb.classes_, index=data.index))
        data = data.drop([name], axis=1)
        return data, lb
    data = data.join(pd.DataFrame(lb.transform(data[name]), columns=lb.classes_, index=data.index)).drop([name], axis=1)
    return data


def replace_nan(data):
    idx, idy = np.where(pd.isnull(data))
    result = np.column_stack([data.index[idx], data.columns[idy]])
    print(result)
    for row, col in result:
        values_array = data[col].dropna().to_numpy()
        # value = statistics.mean(values_array)
        value = statistics.median(values_array)
        data.iloc[row, data.columns.get_loc(col)] = value


def regression_fill_nan(data):
    label_encoding(data, 'region')
    X = []
    y = []

    # izdvojiti sve redove koje nemaju NaN
    for index, row in data.dropna().iterrows():
        X.append([row['oil'], row['region']])
        y.append(row['infant'])

    X = np.array([x for x in X])
    reg = LinearRegression().fit(X, y)

    # prodji kroz sve elemente i gde je NaN tu pozovi predikciju
    for index, row in data.iterrows():
        if pd.isnull(row['infant']):
            train_data.iloc[index, data.columns.get_loc('infant')] = reg.predict(np.array([[row['oil'], row['region']]]))[0]


def calculate_f1_score(y_true, y_predict):
    return f1_score(y_true, y_predict, average='micro')


def standard_scaler(data, scaler=None):
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    # data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return scaled_data, scaler #data, scaler


def min_max_scaler(data, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    # data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return scaled_data, scaler


def cross_validation(X, Y):
    params = {
        "n_components": [4],
        "n_init": [1, 10, 20],
        "max_iter": [1000,  100000],
        "covariance_type": ['diag', 'full', 'tied', 'spherical'],
        "init_params": ['kmeans', "random"]
    }

    gm = GaussianMixture()

    grid = GridSearchCV(estimator=gm, param_grid=params, cv=5, scoring="v_measure_score")
    grid.fit(X, Y)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)


def train_ensemble(X_train, Y_train):
    # najbolji rezultat koji sam dobio: 0.7741935483870968 na test_preview
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8)
    clf.fit(X_train, Y_train)
    return clf


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # print(train_data)
    # print(test_data)

    replace_nan(train_data)
    print(len(train_data))
    train_data = train_data.dropna()
    print(len(train_data))

    # col_names = ['race', 'jobclass', 'health', 'health_ins']
    col_names = ['jobclass', 'health', 'health_ins']
    for name in col_names:
        le = label_encoding(train_data, name)
        label_encoding(test_data, name, le)

    col_names = ['maritl', 'education']
    for name in col_names:
        train_data, ohe = one_hot_encoding(train_data, name)
        test_data = one_hot_encoding(test_data, name, ohe)

    y_train = train_data['race'].to_numpy()
    del train_data['race']
    y_test = test_data['race'].to_numpy()
    del test_data['race']

    train_data, scaler = min_max_scaler(train_data)
    test_data, _ = min_max_scaler(test_data, scaler)

    # PCA
    pca = PCA(svd_solver='full', n_components=10, copy=True)
    pca.fit(train_data)
    print(pca.explained_variance_ratio_)

    # KernelPCA
    # pca = KernelPCA(n_components=5)
    # pca.fit(train_data)

    train_data = pd.DataFrame(data=pca.transform(train_data))
    test_data = pd.DataFrame(data=pca.transform(test_data))
    
    # print(train_data)
    # print(test_data)

    x_train = train_data.to_numpy()

    # svm = SVC(gamma='scale', C=1)
    # svm.fit(x_train, y_train)

    # bgc = BaggingClassifier(n_estimators=10000)
    # bgc.fit(x_train, y_train)
    # ensemble = train_ensemble(x_train, y_train)
    ab = AdaBoostClassifier(n_estimators=100)
    ab.fit(x_train, y_train)

    x_test = test_data.to_numpy()

    # y_predict = svm.predict(x_test)
    y_predict = ab.predict(x_test)
    # y_predict = ensemble.predict(x_test)

    score = calculate_f1_score(y_test, y_predict)
    print(score)

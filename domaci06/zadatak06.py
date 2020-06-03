import sys
# import random
import numpy as np
import pandas as pd
import statistics
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import shapiro
# from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import GradientBoostingClassifier


def label_encoding(data, name, le=None):
    if le is None:
        le = LabelEncoder()
        le = le.fit(list(set(data[name])))
    data[name] = le.transform(data[name])
    return le


def replace_nan(data, col_name):
    value = None
    index = data[col_name].index[data[col_name].apply(np.isnan)]
    for i in index:
        if value is None:
            values_array = data[col_name].dropna().to_numpy()
            # value = statistics.mean(values_array)
            value = statistics.median(values_array)
        data.iloc[i, data.columns.get_loc(col_name)] = value


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
    data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return data, scaler


def min_max_scaler(data, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return data, scaler


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

    train_data = train_data.dropna()

    col_names = ['maritl', 'education', 'race', 'jobclass', 'health', 'health_ins' ]
    for name in col_names:
        le = label_encoding(train_data, name)
        label_encoding(test_data, name, le)

    # print(train_data)
    # print(test_data)

    y_train = train_data['race'].to_numpy()
    del train_data['race']
    y_test = test_data['race'].to_numpy()
    del test_data['race']

    # PCA
    # pca = PCA(n_components=6, svd_solver='full')
    # pca.fit(train_data)

    # KernelPCA
    pca = KernelPCA(n_components=5)
    pca.fit(train_data)


    train_data = pd.DataFrame(data=pca.transform(train_data))
    test_data = pd.DataFrame(data=pca.transform(test_data))
    
    # print(train_data)
    # print(test_data)

    x_train = train_data.to_numpy()

    # svm = SVC()
    # svm.fit(x_train, y_train)

    ensemble = train_ensemble(x_train, y_train)

    x_test = test_data.to_numpy()

    # y_predict = svm.predict(x_test)
    y_predict = ensemble.predict(x_test)

    score = calculate_f1_score(y_test, y_predict)
    print(score)

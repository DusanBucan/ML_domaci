import sys
import random

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV


def statistics_infant(train_data):
    data = train_data.dropna();
    median = []
    mean = []
    mini = []
    maksi = []
    all = []
    for i in ['Asia', 'Africa', 'Americas', 'Europe']:
        x = []
        for row in data.iterrows():
            if row[1][2] == i:
                x.append(row[1][0])
        all += x
        x.sort()
        print(x)
        print(statistics.median(x))
        print(statistics.mean(x))
        print(min(x))
        print(max(x))
        median.append(statistics.median(x))
        mean.append(statistics.mean(x))
        mini.append(min(x))
        maksi.append(max(x))

    median.append(statistics.median(all))
    mean.append(statistics.mean(all))
    mini.append(min(all))
    maksi.append(max(all))


    print('Asia', 'Africa', 'Americas', 'Europe', 'all')
    print(median)
    print(mean)
    print(mini)
    print(maksi)


    # Analize oil Values
    oil_values = [val for val in data['oil']]
    has_oil = len([x for x in oil_values if x == "yes"])
    print("\n Analize oil values")
    print("has oil: ", has_oil, " hasn't oil: ", len(oil_values) - has_oil)


def label_encoding(data, name, le=None):
    if le is None:
        le = LabelEncoder()
        le = le.fit(list(set(data[name])))
    data[name] = le.transform(data[name])
    return le


def replace_nan_by_region(data):
    values = {}
    index = data['infant'].index[data['infant'].apply(np.isnan)]
    for i in index:
        region_name = data['region'][i]
        if region_name not in values:
            values_array = data[data['region'] == region_name]['infant'].dropna().to_numpy()
            # values[region_name] = statistics.mean(values_array)
            values[region_name] = statistics.median(values_array)
        data.iloc[i, data.columns.get_loc('infant')] = values[region_name]


def replace_nan_all(data):
    value = None
    index = data['infant'].index[data['infant'].apply(np.isnan)]
    for i in index:
        if value is None:
            values_array = data['infant'].dropna().to_numpy()
            # value = statistics.mean(values_array)
            value = statistics.median(values_array)
        data.iloc[i, data.columns.get_loc('infant')] = value


def data_preprocessing(train, test):
    le = label_encoding(train, 'oil')
    label_encoding(test, 'oil', le)
    replace_nan_by_region(train)
    # replace_nan_all(train)


def calculate_v_measure_score(Y_test, Y_predict):
    return v_measure_score(Y_test, Y_predict)

def standardScaler(data, scaler):
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return data, scaler

def minMaxScaler(data, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return data, scaler

def show_3D_plot(data):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = plt.axes(projection='3d')

    # for i in set(data['region']):
    #     x_values = data[data['region'] == i]['income'].dropna().to_numpy()
    #     y_values = data[data['region'] == i]['infant'].dropna().to_numpy()
    #     z_values = data[data['region'] == i]['oil'].dropna().to_numpy()
    #     ax.scatter3D(x_values, y_values, z_values, cmap='Greens', label=i)

    regions = [['Americas', 'Africa'], ['Asia', 'Europe']]
    for i in regions:
        x_values = pd.concat([data[data['region'] == i[0]]['income'].dropna(), data[data['region'] == i[1]]['income'].dropna()]).to_numpy()
        y_values = pd.concat([data[data['region'] == i[0]]['infant'].dropna(), data[data['region'] == i[1]]['infant'].dropna()]).to_numpy()
        z_values = pd.concat([data[data['region'] == i[0]]['oil'].dropna(), data[data['region'] == i[1]]['oil'].dropna()]).to_numpy()
        ax.scatter(x_values, y_values, z_values, cmap='Greens', label=i)
    # ax.legend([scatter1_proxy, scatter2_proxy], ['label1', 'label2'], numpoints=1)
    # da vidimo u 2D da li ima neke strukture (elipsa, kruznica, krst)
    # posto samo 10% njih ima naftu..
    # z_zeros = [0 for x in range(0, len(x_values))]
    # ax.scatter3D(x_values, y_values, z_zeros, cmap='Greens')

    ax.set_xlabel('income')
    ax.set_ylabel('infant')
    ax.set_zlabel('oil')
    ax.legend()

    plt.show()


def cross_validation(X, Y):
    params = {
        "n_components": [4],
        "max_iter": [10000],
        "covariance_type": ['diag', 'full', 'tied', 'spherical'],
        "init_params": ['kmeans']
    }

    gm = GaussianMixture()

    grid = GridSearchCV(estimator=gm, param_grid=params, cv=10, scoring="v_measure_score")
    grid.fit(X, Y)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)


def gaussian_mixture(train_data, group1=['Africa', 'Americas'], group2=['Europe', 'Asia']):
    Y_train_group1 = pd.concat([train_data[train_data['region'] == group1[0]]['region'],
                                train_data[train_data['region'] == group1[1]]['region']]).to_numpy()
    Y_train_group2 = pd.concat([train_data[train_data['region'] == group2[0]]['region'],
                                train_data[train_data['region'] == group2[1]]['region']]).to_numpy()
    Y_train_all = np.array([1]*len(Y_train_group1) + [2]*len(Y_train_group2))

    X_train_group1 = pd.concat([train_data[train_data['region'] == group1[0]],
                                train_data[train_data['region'] == group1[1]]])
    X_train_group2 = pd.concat([train_data[train_data['region'] == group2[0]],
                                train_data[train_data['region'] == group2[1]]])

    del X_train_group1['region']
    del X_train_group2['region']

    X_train_group1 = X_train_group2.to_numpy()
    X_train_group2 = X_train_group2.to_numpy()
    X_train_all = np.concatenate((X_train_group1, X_train_group2))

    gm_group1 = GaussianMixture(n_components=2, max_iter=20000)
    gm_group2 = GaussianMixture(n_components=2, max_iter=20000)
    gm_all = GaussianMixture(n_components=2, max_iter=100000, covariance_type='diag', n_init=10, init_params='random')
    gm_group1.fit(X_train_group1, Y_train_group1)
    gm_group2.fit(X_train_group2, Y_train_group2)
    gm_all.fit(X_train_all, Y_train_all)

    return gm_group1, gm_group2, gm_all


def predict_gm(gm_group1, gm_group2, gm_all, X_test):
    Y_predict_all = gm_all.predict(X_test)
    print(Y_predict_all)
    return 'cao'


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # statistics_infant(train_data)

    data_preprocessing(train_data, test_data)

    # brisanje redova sa nan vrednostima
    # train_data = train_data.dropna()
    # train_data.reset_index(drop=True, inplace=True)

    # show_3D_plot(train_data)

    # podeli se u dve grupe, jedan klasifikator da prepozna kojoj grupi pripada, druga dva da prepozna kom regionu
    # unutar grupe pripada
    # radi lose, ne prepozna dobro kojoj grupi pripada, ako hocete probajte
    # gm_group1, gm_group2, gm_all = gaussian_mixture(train_data)
    #
    # Y_test = test_data['region'].to_numpy()
    # del test_data['region']
    # X_test = test_data.to_numpy()
    #
    # Y_predict = predict_gm(gm_group1, gm_group2, gm_all, X_test)


    Y_train = train_data['region'].to_numpy()
    del train_data['region']
    X_train = train_data.to_numpy()

    # X_train, scaler = minMaxScaler(X_train, None)

    # cross_validation(X_train, Y_train)

    weights = [0.298, 0.265, 0.194, 0.243]
    gm = GaussianMixture(n_components=4, covariance_type='diag', max_iter=100000, n_init=100, weights_init=weights)
    gm.fit(X_train, Y_train)
    # ne znam da li ovo sve ovako
    gm.weights_ = weights
    print(gm.weights_)

    Y_test = test_data['region'].to_numpy()
    del test_data['region']
    X_test = test_data.to_numpy()
    # X_test, scaler = minMaxScaler(X_test, scaler)

    Y_predict = gm.predict(X_test)
    score = calculate_v_measure_score(Y_test, Y_predict)
    print(score)


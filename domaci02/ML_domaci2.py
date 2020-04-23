import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


def calculate_rmse(y_true, y_predict):
    N = len(y_true)
    sum = 0
    for i in range(N):
        sum += (y_predict[i] - y_true[i]) ** 2
    rmse = math.sqrt(sum / N)
    return rmse


def view_data(train_data, test_data):
    # print(train_data['zvanje'])
    print(test_data)
    plt.scatter(train_data['zvanje'], train_data['plata'])
    plt.show()
    plt.scatter(train_data['oblast'], train_data['plata'])
    plt.show()
    plt.scatter(train_data['godina_doktor'], train_data['plata'])
    plt.show()
    plt.scatter(train_data['godina_iskustva'], train_data['plata'])
    plt.show()
    plt.scatter(train_data['pol'], train_data['plata'])
    plt.show()
    x_data = [row[1][4] for row in train_data.iterrows() if row[1][0] == 'Prof']
    y_data = [row[1][5] for row in train_data.iterrows() if row[1][0] == 'Prof']

    plt.scatter(x_data, y_data)
    plt.show()


def one_hot_encoding(data, name):
    return pd.get_dummies(data, columns=[name])


def label_encoding(data, name):
    data[name] = data[name].astype('category').cat.codes
    return data


def categorical_data(data, fun):
    data = fun(data, "zvanje")
    data = fun(data, "oblast")
    data = fun(data, "pol")
    return data


def normalization(data, d_norm):
    for col in data:
        d = d_norm[col]
        for idx, row in enumerate(data[col]):
            data.at[idx, col] = row/d
    return data


def d_normalization(data):
    d = {}
    for col in data:
        column_data = data[col].to_numpy()
        d[col] = (math.sqrt(sum(column_data**2)))
    return d


def mean_val_nor(data):
    mean_val = []
    for col in data:
        mean_val.append(np.mean(np.asanyarray(data[col], dtype=np.float64)))
    return mean_val


def std_val_nor(data):
    std_val = []
    for col in data:
        std_val.append(np.std(np.asanyarray(data[col], dtype=np.float64)))
    return std_val


def z_score_normalization(data,  mean_val, std_val):
    for i, col in enumerate(data):
        for idx, row in enumerate(data[col]):
            data.at[idx, col] = (row - mean_val[i]) / std_val[i]
    return data


def predict(x, theta):
    return sum(x[i] * theta[i] for i in range(len(theta)))


def lasso_coordinate_descent(x, y, step=0.1, l=5):
    N = len(x)
    D = len(x[0])
    theta = [0.0] * D
    theta = np.asarray(theta, dtype=np.float64)
    while True:
        old_theta = theta.copy()
        for j in range(D):
            r = 0
            for i in range(N):
                x_predict = x[i].copy()
                x_predict[j] = 0
                k = predict(x_predict, theta)
                r += x[i][j] * (y[i] - k)

            if r < l/2:
                theta[j] = r + l/2
            elif r > l/2:
                theta[j] = r - l/2
            else:
                theta[j] = 0
        if sum(abs(theta - old_theta)) < step:
            break
    return theta


def ridge(x, y, alpha=0.001, max_iters=1000, l=1):
    N = len(x)
    D = len(x[0])
    theta = [1.0] * D
    theta = np.asarray(theta, dtype=np.float64)

    for t in range(max_iters):
        for j in range(D):
            sum = 0
            for i in range(N):
                y_predict = predict(x[i], theta)

                sum += (y_predict - y[i]) * x[i][j]
                theta[j] = theta[j] * (1 - alpha * l) - alpha / N * sum

    return theta


if __name__ == '__main__':
    trainPath = 'dataset/train.csv'
    testPath = 'dataset/test_preview.csv'

    train_data = pd.read_csv(trainPath)
    test_data = pd.read_csv(testPath)

    # view_data(train_data, test_data)
    #label encodin i one hot encoding
    train_data = categorical_data(train_data, label_encoding)

    print(train_data.columns.values)
    train_data = train_data.astype('float64')
    y_train = train_data['plata'].to_numpy()
    del train_data['plata']

    # normalizadija i ridge
    # mean_val = mean_val_nor(train_data)
    # std_val = std_val_nor(train_data)
    # train_data = z_score_normalization(train_data, mean_val, std_val)
    # x_train = train_data.to_numpy()
    # theta = ridge(x_train, y_train)
    # print(theta)


    #normailzacija i lasso
    d_norm = d_normalization(train_data)
    train_data = normalization(train_data, d_norm)
    x_train = train_data.to_numpy()
    theta = lasso_coordinate_descent(x_train, y_train, step=0.001, l=1)
    print(theta)

    ## iteracije da se utvred koja obelezja ne trebaju
    # x = []
    # y = []
    # for i in range(1, 100000, 100):
    #     x.append(i)
    #     y.append(lasso_coordinate_descent(x_train,y_train, l=i))
    # y = np.array(y).transpose()
    # print(y)
    # x = [x]*len(y)
    # print(x)
    # for i in range(len(y)):
    #     plt.plot(x[i], y[i], i)
    # plt.show()

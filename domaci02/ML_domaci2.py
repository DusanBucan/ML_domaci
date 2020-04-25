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
    if name == "zvanje":
        data[name] = [0 if d == "Prof" else (1 if d == "AssocProf" else 2) for d in data[name]]
    elif name == "pol":
        data[name] = [1 if d == "Male" else 0 for d in data[name]]
    elif name == "oblast":
        data[name] = [1 if d == "A" else 0 for d in data[name]]
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
            data.at[idx, col] = row / d
    return data


def d_normalization(data):
    d = {}
    for col in data:
        column_data = data[col].to_numpy()
        d[col] = (math.sqrt(sum(column_data ** 2)))
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


def z_score_normalization(data, mean_val, std_val):
    for i, col in enumerate(data):
        if col == "pol" or col == "oblast":  # fora je su samo 0 ili 1 pa ne treba
            continue
        for idx, row in enumerate(data[col]):
            data.at[idx, col] = (row - mean_val[i]) / std_val[i]
    return data


def min_values(data):
    min_val = []
    for col in data:
        min_val.append(np.min(np.asanyarray(data[col], dtype=np.float64)))
    return min_val


def max_values(data):
    max_val = []
    for col in data:
        max_val.append(np.max(np.asanyarray(data[col], dtype=np.float64)))
    return max_val


def min_max_normalization(data, mins, maxs):
    for i, col in enumerate(data):
        if col == "pol" or col == "oblast":  # fora je su samo 0 ili 1 pa ne treba
            continue
        for idx, row in enumerate(data[col]):
            data.at[idx, col] = (row - mins[i]) / maxs[i] - mins[i]
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

            if r < l / 2:
                theta[j] = r + l / 2
            elif r > l / 2:
                theta[j] = r - l / 2
            else:
                theta[j] = 0
        if sum(abs(theta - old_theta)) < step:
            break
    return theta


def ridge(x, y, alpha=0.0483, max_iters=500, l=0.000001):
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

def ridge2(x, y, alpha=0.05, step=0.1, l=0.1):
    N = len(x)
    D = len(x[0])
    theta = [1.0] * D
    theta = np.asarray(theta, dtype=np.float64)

    while True:
        old_theta = theta.copy()
        for j in range(D):
            suma = 0
            for i in range(N):
                y_predict = predict(x[i], theta)
                suma += (y_predict - y[i]) * x[i][j]
            theta[j] = theta[j] * (1 - alpha * l) - alpha / N * suma
        if sum(abs(theta - old_theta)) < step:
            break
    return theta

""""
    vrati PROSECAN RMSE
    
    NE MENJA PARAMETRE KOD RIDGE
    NE BIRA NAJBOLJE TETA

"""
def train_validation(train_data, test_data, size=10):
    groups_x = []
    groups_y = []
    err = []

    min_err = math.inf
    best_theta = []

    train_data.sort_values(by=['plata'], inplace=True)
    train_data = train_data.astype('float64')
    y_train = train_data['plata'].to_numpy()
    del train_data['plata']
    min_val = min_values(train_data)
    max_val = max_values(train_data)
    train_data = min_max_normalization(train_data, min_val, max_val)
    x_train = train_data.to_numpy()

    n = 0
    for i in range(len(y_train)):
        if len(groups_x) <= n:
            groups_x.append([])
            groups_y.append([])
        groups_x[n].append(x_train[i])
        groups_y[n].append(y_train[i])

        if n == size - 1:
            n = 0
        else:
            n += 1
    for i in range(0, size):
        x_t = []
        y_t = []
        for j in range(0, size):
            if i != j:
                x_t += groups_x[j]
                y_t += groups_y[j]

        theta = ridge2(np.asarray(x_t, dtype=np.float64), np.asarray(y_t, dtype=np.float64))

        x = []
        for j in groups_x[i]:
            x.append(predict(j, theta))
        err.append(calculate_rmse(groups_y[i], x))
        print(err)
    print(sum(err)/len(err))



# statistic utill function
def make_bar_chart(data, column_name, no_bars):
    values = np.asanyarray(data[column_name], dtype=np.float64)
    values = np.sort(values)
    min_val = np.min(values)
    max_val = np.max(values)
    result = []
    bars_values = []

    bar_size = (max_val - min_val) / no_bars
    val = min_val
    while val < max_val:
        a = [v for v in values if val <= v < val + bar_size]
        result.append(len(a))
        bars_values.append(val)
        val += bar_size

    total = np.sum(result)
    print(result)
    print(total)
    print(len(values))

    plt.bar(bars_values, result, align='center', alpha=0.5)

    plt.show()


def make_quartile_plot(data, column_name):

    values = np.asanyarray(data[column_name], dtype=np.float64)
    plt.boxplot(values)
    for v in values:
        plt.plot(1, v, 'r.', alpha=0.4)
    plt.show()



if __name__ == '__main__':
    trainPath = 'dataset/train.csv'
    testPath = 'dataset/test_preview.csv'

    train_data = pd.read_csv(trainPath)
    test_data = pd.read_csv(testPath)


    # make_quartile_plot(train_data, "godina_iskustva")
    # make_bar_chart(train_data, "godina_iskustva", 5)

    view_data(train_data, test_data)
    
    # # #label encoding
    # train_data = categorical_data(train_data, label_encoding)
    # test_data = categorical_data(test_data, label_encoding)
    #
    # # print(train_data.columns.values)
    # # train_validation(train_data, test_data)
    #
    # train_data = train_data.astype('float64')
    # y_train = train_data['plata'].to_numpy()
    # del train_data['plata']
    # # zasto su izbacena ova 2?
    # # del train_data['godina_doktor']
    # # del train_data['godina_iskustva']
    #
    # # print(train_data[0:10])
    #
    # # # Z-SCORE i ridge
    # # mean_val = mean_val_nor(train_data)
    # # std_val = std_val_nor(train_data)
    # # train_data = z_score_normalization(train_data, mean_val, std_val)
    #
    # # MIN-MAX normalizacija i ridge
    # min_val = min_values(train_data)
    # max_val = max_values(train_data)
    # train_data = min_max_normalization(train_data, min_val, max_val)
    #
    #
    # x_train = train_data.to_numpy()
    # theta = ridge(x_train, y_train)
    # print(theta)
    #
    # test_data = test_data.astype('float64')
    # y_test = test_data['plata'].to_numpy()
    # del test_data['plata']
    # # del test_data['godina_doktor']
    # # del test_data['godina_iskustva']
    #
    # test_data = min_max_normalization(test_data, min_val, max_val)
    # x_test = test_data.to_numpy()
    # x = []
    # for i in x_test:
    #     x.append(predict(i, theta))
    # print(calculate_rmse(y_test, x))
    
    #normailzacija i lasso
    # d_norm = d_normalization(train_data)
    # train_data = normalization(train_data, d_norm)
    # x_train = train_data.to_numpy()
    # theta = lasso_coordinate_descent(x_train, y_train, step=0.001, l=1)
    # print(theta)

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

    # iteracije da se utvrdi alpha za ridge
    # x = []
    # y = []
    # z = []
    # for i in np.linspace(0.01, 0.2, 10):
    #     x.append(i)
    #     theta = ridge(x_train, y_train, alpha=i, l=0)
    #     r = []
    #     for j in x_test:
    #         r.append(predict(j, theta))
    #     rmse = calculate_rmse(y_test, r)
    #     y.append(rmse)
    #     r = []
    #     for j in x_train:
    #         r.append(predict(j, theta))
    #     rmse = calculate_rmse(y_test, r)
    #     z.append(rmse)
    #     print(i, rmse)
    # plt.plot(x, y)
    # plt.plot(x, z)
    # plt.show()
    # print(x, y, z)
    # # 0.0483
    #
    # ## iteracije da se utvrdi lambda za ridge
    # x = []
    # y = []
    # for i in np.linspace(0, 10, 20):
    #     x.append(i)
    #     theta = ridge(x_train, y_train, alpha=0, l=i)
    #     r = []
    #     for j in x_test:
    #         r.append(predict(j, theta))
    #     rmse = calculate_rmse(y_test, r)
    #     y.append(rmse)
    #     print(i, rmse)
    # plt.plot(x, y)
    # plt.show()

import sys

import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math


def calculate_rmse(y_true, y_predict):
    rmse = 0
    N = len(y_true)
    sum = 0
    for i in range(N):
        sum += (y_predict[i] - y_true[i]) ** 2
    rmse = math.sqrt(sum / N)
    return rmse

# metode za predprocesiranje podataka

def remove_outliers(data):
    filtered_data = []
    for d in data:
        if not((d[0] < 3100 and d[1] > 1450) or (d[0] > 4450 and d[1] < 1110) or (d[0] > 4900)):
            filtered_data.append(d)

    return np.asarray(filtered_data, dtype=np.uint16)

def remove_outliers_by_z_score(data):

    # ako za svaki izracunam odnos x/y i y/x zavisi
    # i onda nadjem za to Z-score i pobrisem tu krajeve
    # onda ce da mi ostanu podaci samo na

    data_without_outliers = []
    x_y_ration = []
    for d in data:
        score = 0
        if d[0] > d[1]:
            score = d[0]/d[1]
        else:
            score = d[1] / d[0]
        x_y_ration.append(score)

    x_data = [i[0] for i in data]
    y_data = [i[1] for i in data]

    # check_gaussian_dsitrubution(x_y_ration)

    mean_val_x = np.mean(np.asanyarray(x_y_ration, dtype=np.float64))
    std_val_x = np.std(np.asanyarray(x_y_ration, dtype=np.float64))

    # print(mean_val_x)
    # print(std_val_x)


    #imas sada izracunate Z_score-ove
    x_data_z_scores = z_score(x_y_ration, mean_val_x, std_val_x)

    check_gaussian_dsitrubution(x_data_z_scores)
    #
    # print(np.max(x_data_z_scores))
    # print(np.min(x_data_z_scores))

    for i in range(len(x_data_z_scores)):
        if abs(x_data_z_scores[i]) <= 1.42:
            data_without_outliers.append(data[i])

    print(len(data_without_outliers))
    return np.asarray(data_without_outliers, dtype=np.int16)

def z_score(data,  mean_val, std_val):
    return [(d - mean_val) / std_val for d in data]

# da proveri da li values ima gausuvo raspodelu
def check_gaussian_dsitrubution(values):

    max = np.max(values)
    min = np.min(values)

    mapa = {}
    retval = []
    interval_size = (max-min)/10
    print(max)
    print(min)
    print(interval_size)
    print("------------")
    intervl_start = min
    while intervl_start < max:
        a = []
        for v in values:
            if v >= intervl_start and v < intervl_start + interval_size:
                a.append(v)
        retval.append(((intervl_start, intervl_start + interval_size), len(a)))
        intervl_start += interval_size

    x = [c[0][1] for c in retval]
    y = [c[1] for c in retval]
    plt.bar(x, y)
    plt.show()

    print(retval)

    # mean_val_x = np.mean(np.asanyarray(values, dtype=np.float64))
    # std_val_x = np.std(np.asanyarray(values, dtype=np.float64))


def stratification(data, test_size=0.1):
    NO_GROUPS = 10
    # da podeli na 10 grupa pa onda u zavisnosti koliko ide test toliko grupa ce da spoji za test
    # npr ako je test=0.3 spojice 3 grupe za test.
    test_size = int(test_size * 10)
    train_data = []
    test_data = []

    data = remove_outliers(data)  # obrise outlier-e
    data = data[data[:, 1].argsort()]

    # kljuc je id grupe 0,1,2,3... , value je lista paraova (x,y)
    data_groups = {}

    key_id = 0
    for d in data:
        if key_id not in data_groups:
            data_groups[key_id] = []

        data_groups[key_id].append(d)
        key_id = key_id + 1 if key_id < (NO_GROUPS - 1) else 0

    # izmesaj redosled kljuceva
    a1 = list(data_groups.items())
    np.random.shuffle(a1)
    data_groups = dict(a1)

    i = 0
    for key in data_groups.keys():
        data_group = data_groups[key]
        x = [ds[0] for ds in data_group]
        y = [ds[1] for ds in data_group]
        if i < test_size:
            for d in data_group:
                test_data.append(d)
        else:
            for d in data_group:
                train_data.append(d)
        i += 1
    return np.asarray(train_data, dtype=np.int64), np.asarray(test_data, dtype=np.int64)



def predict(x, theta):
    return sum(x**i * theta[i] for i in range(len(theta)))


# =============== metode za izracunavanje parametara =========
def fit_batch_gd_mse(x, y, D = 2, max_iters = 10000, alpha=0.001, l = 0):
    N = len(x)
    theta = [1.0] * D
    theta = np.asarray(theta, dtype=np.float64)

    for t in range(max_iters):
        for d in range(D):
            sum = 0
            for i in range(N):
                y_predict = predict(x[i], theta)

                # ovo je least mean squares iz njega izvod ---> prezentacija 3
                if not math.isnan(y_predict):
                    sum += (y_predict - y[i]) * x[i] ** d
            if d == 0:
                theta[d] = theta[d] - (alpha / N) * sum
            else:
                theta[d] = theta[d] * (1 - l * alpha) - (alpha / N) * sum

    return theta


def fit_normal_equasion(x, y, D=2):
    # matrica dimenzije N x D
    x = [[i ** d for d in range(D)] for i in x]
    # transponovana x matrica
    x_t = np.transpose(x)

    # pomnozene matrice x_t i x
    x_t_x = np.dot(x_t, x)

    # inverzna matrica od pomnozenih matrica x_t i x
    x_t_x_inv = np.linalg.inv(x_t_x)

    # pomnozene matrice x_t_x_inv i x_t
    x_t_x_inv_x_t = np.dot(x_t_x_inv, x_t)

    # pomnozene matrice x_t_x_inv_x_t i y, dobijeno theta
    theta = np.dot(x_t_x_inv_x_t, y)
    return theta


if __name__ == '__main__':

    trainPath = sys.argv[1]
    testPath = sys.argv[2]

    train_data = pd.read_csv(trainPath)
    test_data = pd.read_csv(testPath)

    test_data = test_data.to_numpy()
    train_data = train_data.to_numpy()

    train = remove_outliers_by_z_score(train_data)
    #
    # train, test = stratification(train_data, test_size=0)
    #
    x_train = [i[0] for i in train]
    y_train = [i[1] for i in train]

    # plt.scatter(x_train, y_train)



    mean_val_x = np.mean(np.asanyarray(x_train, dtype=np.float64))
    std_val_x = np.std(np.asanyarray(x_train, dtype=np.float64))

    x_train = z_score(x_train, mean_val_x, std_val_x)


    # theta = fit_batch_gd_mse(x_train, y_train, 2, 40000, 0.001)
    theta = fit_normal_equasion(x_train, y_train, 2)

    x_test = [i[0] for i in test_data]
    y_test = [i[1] for i in test_data]

    x_test = z_score(x_test, mean_val_x, std_val_x)
    y_predict = [predict(x, theta) for x in x_test]

    x = np.linspace(2000, 5000, 1000)
    x_z = z_score(x, mean_val_x, std_val_x)
    y_pred = [predict(t, theta) for t in x_z]
    # plt.plot(x, y_pred)
    # plt.show()
    print(calculate_rmse(y_test, y_predict))

import sys

# import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
# from scipy.stats import shapiro, normaltest
# from sklearn.linear_model import ElasticNet
# from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true, y_predict):
    N = len(y_true)
    sum = 0
    for i in range(N):
        sum += (y_predict[i] - y_true[i]) ** 2
    rmse = math.sqrt(sum / N)
    return rmse

#
# def view_data(train_data, test_data):
#     # print(train_data['zvanje'])
#     print(test_data)
#     plt.scatter(train_data['zvanje'], train_data['plata'])
#     plt.show()
#     plt.scatter(train_data['oblast'], train_data['plata'])
#     plt.show()
#     plt.scatter(train_data['godina_doktor'], train_data['plata'])
#     plt.show()
#     plt.scatter(train_data['godina_iskustva'], train_data['plata'])
#     plt.show()
#     plt.scatter(train_data['pol'], train_data['plata'])
#     plt.show()
#     x_data = [row[1][4] for row in train_data.iterrows() if row[1][0] == 'Prof']
#     y_data = [row[1][5] for row in train_data.iterrows() if row[1][0] == 'Prof']
#
#     plt.scatter(x_data, y_data)
#     plt.show()


def one_hot_encoding(data, name):

    data = pd.get_dummies(data, columns=[name])

    #dopunjaavanje dataFrame-a
    if "pol_Male" not in data and "pol" == name:
        data['pol_Male'] = 0
    if "pol_Female" not in data and "pol" == name:
        data['pol_Female'] = 0
    if "oblast_A" not in data and "oblast" == name:
        data['oblast_A'] = 0
    if "oblast_B" not in data and "oblast" == name:
        data['oblast_B'] = 0


    return data


def label_encoding(data, name):

    # za zvanje ide Ordinal Encoding
    if name == "zvanje":
        data[name] = [3 if d == "Prof" else (2 if d == "AssocProf" else 1) for d in data[name]]
    elif name == "pol":
        data[name] = [1 if d == "Male" else 0 for d in data[name]]

    # treba da ide cist one-hot-encoding..
    elif name == "oblast":
        # bar su svi koeficijenti pozitivni kad se ovako okrene...
        data[name] = [0 if d == "A" else 1 for d in data[name]]
    return data


def categorical_data(data, fun):
    data = fun(data, "zvanje")
    # data = fun(data, "oblast")
    data = one_hot_encoding(data, "oblast")
    data = one_hot_encoding(data, "pol")
    # data = fun(data, "pol")

    if "bias" not in data:
        data['bias'] = 1

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
        if col == "pol_Male" or\
                col == "oblast_A" or col == "oblast_A"\
                or col == "pol_Female" or col == "bias":
            continue
        if  col == "zvanje" or col == "godina_iskustva":
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
        if col == "pol_Male" or\
                col == "oblast_A" or col == "oblast_A"\
                or col == "pol_Female" or col == "bias":  # fora je su samo 0 ili 1 pa ne treba
            continue

        for idx, row in enumerate(data[col]):
            # maxs[i] - mins[i] ---> nije bilo u zagradi...
            data.at[idx, col] = (row - mins[i]) / (maxs[i] - mins[i])
    return data


def predict(x, theta):
    return sum(x[i] * theta[i] for i in range(len(theta)))


def lasso_coordinate_descent(x, y, step=0.1, l=5, max_inter=5000):
    N = len(x)
    D = len(x[0])
    theta = [0.0] * D
    theta = np.asarray(theta, dtype=np.float64)
    iterations = 0
    while True:
        iterations += 1
        old_theta = theta.copy()
        for j in range(D):
            r = 0
            for i in range(N):
                x_predict = x[i].copy()
                x_predict[j] = 0
                # kod svakog primera iz skupa je ova koordinata zarznuta
                # k ce biti predikcija BEZ KOORDINATE za koju odredjujemo TETA
                k = predict(x_predict, theta)
                r += x[i][j] * (y[i] - k)

            if r < (l / 2):
                theta[j] = r + l / 2
            elif r > (l / 2):
                theta[j] = r - l / 2
            else:
                theta[j] = 0
        if sum(abs(theta - old_theta)) < step:
            break
        if iterations >= max_inter:
            break
    return theta


# 0.000001 --> 1545.468
# 0.00001 --> 15455.98
# 0.0001 ---> 15461
# 1 --> 50k ===> veliko sistemsko odsupanje, ali SMANJE SE KOEFICIJENTI
def ridge(x, y, alpha=0.0483, max_iters=500, l=0.001):
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

    # MOZE da se doda da ODREDJUJE labmda za RIDGE sa UNAKRSNOM...
    # sad za svaku od grupa napravi
    for i in range(0, size):
        x_t = []
        y_t = []
        for j in range(0, size):
            if i != j:
                x_t += groups_x[j]
                y_t += groups_y[j]

        #napravio train i validacioni
        theta = ridge2(np.asarray(x_t, dtype=np.float64), np.asarray(y_t, dtype=np.float64))

        #izracuna RMESE nad validacionim
        for j in groups_x[i]:
            x.append(predict(j, theta))
        err.append(calculate_rmse(groups_y[i], x))



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

    check_normal_dist(values)


def check_normal_dist(data):
    pass
    # # stat, p = normaltest(data)
    # stat, p = shapiro(data)
    # print("============================")
    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    # # interpret
    # alpha = 0.05
    # if p > alpha:
    #     print('Sample looks Gaussian (fail to reject H0)')
    # else:
    #     print('Sample does not look Gaussian (reject H0)')


def make_quartile_plot(data, column_name):
    pass
    # values = np.asanyarray(data[column_name], dtype=np.float64)
    # plt.boxplot(values)
    # for v in values:
    #     plt.plot(1, v, 'r.', alpha=0.4)
    # plt.show()


def remove_outLiers_by_column_name(data, column_name):
    Q1 = data[column_name].quantile(0.15)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1  # IQR is interquartile range.

    #ako je vece od donje granice i manje od gornje onda da
    # ga IZbACI iz PODACIMA???

    # print(Q1 - 1.5 * IQR)

    new_data = data.drop(data[(data[column_name] <= Q1 - 1.5 * IQR) | (data[column_name] >= Q3 + 1.5 * IQR)].index, inplace=False)
    new_data = new_data.reset_index(drop=True)
    return new_data

def show_cov(data):

    cov_matrix = data.cov()
    print(cov_matrix)


def ridgeUgradjeni():
    pass


"""
    sa Min-Max normalizacijom ako TERAS SVE NA POZITIVNO minimalna greska je za alpha = 0.0001 ==> ok 14.8K
            ===> ugura na pol i oblast ono da JEDNA BUDE 0 a druga neki broj, bias = 0

    Ako stavimo fit_intercept = 0 ===> jer imamo nas Bias dodat onda nas bias PODESI na mali broj ali
        greska ostane ista...
        
        
    AKO PREBACIMO NA Z-SCORE GRESKA ista ali SU KOEFICIJENTI MANJI.. ---> uvek stavi godine_iskustva na 0


    ako ga probamo kao:
        LASSO --> kada je l1_ratio = 1 onda radi kao Lasso
        RIDGE --> l1_ratio = 0
        
    
    kada radi kao Lasso i kada sa Min-Max izvaci odredjena obelezja a to su:
        a. bias
        b. pol_female
        c. godine_iskustva
        
    ridge daje SLICNE REZULTATE kao NAS ali MALO BOLJI jer nekako zakuca godine_iskustva na 0...

"""
def elasticNet(x_train, y_train, x_test, y_test):
    # alphas = [0.00001, 0.0001, 0.001, 0.01 ,0.1 ,0.2, 0.7]
    #
    # for a in alphas:
    #     model = ElasticNet(
    #         alpha=a, max_iter=5000, positive=False, fit_intercept=False, l1_ratio=0)\
    #         .fit(x_train, y_train)
    #     score = model.score(x_test, y_test)
    #     pred_y = model.predict(x_test)
    #     mse = mean_squared_error(y_test, pred_y)
    #     print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    #           .format(a, score, mse, np.sqrt(mse)))
    #     print(model.coef_)
    #     print(model.intercept_)
    #     print("\n")
    pass


if __name__ == '__main__':

    trainPath = sys.argv[1]
    testPath = sys.argv[2]


    # trainPath = 'dataset/train.csv'
    # testPath = 'dataset/test_preview.csv'

    train_data = pd.read_csv(trainPath)
    test_data = pd.read_csv(testPath)


    # make_quartile_plot(train_data, "plata")
    # izgleda da godina_iskustva nema normalnu raspodelu...
        # a vidi se i sa grafika.. da nema najvise ljudi
        # sa srednjim brojem godina iskustva nego


    """
        izgleda i godine doktor da nemaju normalnu tako kazu oni testovi..
    
        ni plata?????
    
    """

    # make_bar_chart(train_data, "oblast", 5)

    train_data = remove_outLiers_by_column_name(train_data, "plata")
    # make_quartile_plot(train_data, "godina_doktor")

    # make_bar_chart(train_data, "plata", 5)

    # view_data(train_data, test_data)
    
    #label encoding
    train_data = categorical_data(train_data, label_encoding)
    test_data = categorical_data(test_data, label_encoding)

    # p=0
    # pAssocProf = 0
    # pAc = 0
    # for a in train_data.values:
    #     if a[0] == 3:
    #         p += 1
    #     elif a[0] == 2:
    #         pAssocProf += 1
    #     else:
    #         pAc += 1
    #
    # print(p, "\nAssocProf:", pAssocProf, "\nAssProf", pAc)
    # make_bar_chart(train_data, "zvanje", 3)
    #
    # print(train_data.columns.values)
    #
    #
    # # train_validation(train_data, test_data)
    #
    train_data = train_data.astype('float64')
    y_train = train_data['plata'].to_numpy()

    del train_data['plata']

    # zasto su izbacena ova 2?
    # del train_data['godina_doktor']
    # del train_data['godina_iskustva']

    # print(train_data[0:10])

    # Z-SCORE i ridge
    # mean_val = mean_val_nor(train_data)
    # std_val = std_val_nor(train_data)
    # train_data = z_score_normalization(train_data, mean_val, std_val)

    ## MIN-MAX normalizacija i ridge
    min_val = min_values(train_data)
    max_val = max_values(train_data)
    train_data = min_max_normalization(train_data, min_val, max_val)

    # da pokaze kovarijansu odnosno medjuzavisnost izmedju obelezja
    # show_cov(train_data)

    x_train = train_data.to_numpy()
    theta = ridge(x_train, y_train)
    # print(theta)

    test_data = test_data.astype('float64')
    y_test = test_data['plata'].to_numpy()
    del test_data['plata']
    # del test_data['godina_doktor']
    # del test_data['godina_iskustva']

    # test_data = z_score_normalization(test_data,mean_val,std_val)
    test_data = min_max_normalization(test_data, min_val, max_val)
    x_test = test_data.to_numpy()
    # elasticNet(x_train, y_train, x_test, y_test)
    x = []
    for i in x_test:
        x.append(predict(i, theta))
    print(calculate_rmse(y_test, x))


    #normailzacija i lasso
    # d_norm = d_normalization(train_data)
    # train_data = normalization(train_data, d_norm)
    # x_train = train_data.to_numpy()
    # theta = lasso_coordinate_descent(x_train, y_train, step=0.001, l=40)
    # print(theta)

    ## iteracije da se utvred koja obelezja ne trebaju
    # x = []
    # y = []
    # for i in range(1, 100000, 10000):
    #     x.append(i)
    #     y.append(lasso_coordinate_descent(x_train, y_train, l=i))
    # y = np.array(y).transpose()
    # print(y)
    # x = [x]*len(y)
    # print(x)
    # for i in range(len(y)):
    #     plt.plot(x[i], y[i], i)
    # plt.show()





    # iteracije da se utvrdi alpha za ridge
    # alpha je LEARNING RATE za GradientDecent
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
    #     #     ovde je bilo y_test ---> a treba y_train
    #     rmse = calculate_rmse(y_train, r)
    #     z.append(rmse)
    #     # print(i, rmse)
    # plt.plot(x, y) # RMSE na Test skupu
    # plt.plot(x, z) # RMSE na Train skupu
    # plt.show()
    # for alpha, rmse_test, rmse_train in zip(x, y, z):
    #     print("alpha: ", alpha, " test_rmse:", rmse_test, " train_rmse: ", rmse_train)
    # print(x, y, z)
    # 0.0483 ---> milica dobila
    # 0.0500 ---> kad sa izbacio outilier-e


    # ====================== ODREDJIVANJE LAMBDA KOD RIDGE ++++++++++++++++++++++++++++++++++++=

    # treba namestit

    # ## iteracije da se utvrdi lambda za ridge, fora vec imas postavljen OLS
    # (je l to znaci da alpha treba ovde da bude zadato.. msm ono je odredjeno vec za OLS)
    # ili treba njih 2 u kombinaciji da se odredjuju????

    # na Predavanjima Ridge regularizacija kod grafika tamo gde inace stoji alpha je bio samo 1

    # kako ga odabrati?
    # pa imas u Ridge regularizaciji onaj grafik kad
        #RMSE test PRESTANE da PADA
        #RMSE train raste al da ne BUDE BAS BAS...

    # sta ovde ne valja sto tamo malo l da bude??

    # jbt meni samo rastu greske na test_skupu.
    # OLS ispada najbolji to je kad je l = 0
    # je l to zato sto sam alpha odredio gledajuci
    # grafike za train i test, a ne za train i validacioni?

    # x = []
    # y = []
    # z = []
    # for i in np.linspace(0, 4, 20):
    #     x.append(i)
    #     theta = ridge(x_train, y_train, alpha=0.0483, l=i)
    #     r = []
    #     for j in x_test:
    #         r.append(predict(j, theta))
    #     rmse = calculate_rmse(y_test, r)
    #     y.append(rmse)
    #     r = []
    #     for j in x_train:
    #         r.append(predict(j, theta))
    #     #     ovde je bilo y_test ---> a treba y_train
    #     rmse = calculate_rmse(y_train, r)
    #     z.append(rmse)
    #
    # plt.plot(x, y)  # RMSE na Test skupu
    # plt.plot(x, z) # RMSE na Train skupu
    # plt.show()
    # for alpha, rmse_test, rmse_train in zip(x, y, z):
    #     print("lambda: ", alpha, " test_rmse:", rmse_test, " train_rmse: ", rmse_train)


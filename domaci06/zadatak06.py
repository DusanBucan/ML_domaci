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
from sklearn.neighbors import KNeighborsClassifier


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


def standard_scaler(data, col, scaler=None):
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(data[[col]])
    scaled_data = scaler.transform(data[[col]])
    # data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return scaled_data, scaler #data, scaler


def min_max_scaler(data, col, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler()
        scaler.fit(data[[col]])
    scaled_data = scaler.transform(data[[col]])
    # data = [[scaled_data[index][0], scaled_data[index][1], data[index][2]] for index, d in enumerate(data)]
    return scaled_data, scaler


def cross_validation(X, Y):
    # params = {
    #     "learning_rate": [0.01, 0.1, 0.2],
    #     "n_estimators": [700, 1000, 1500],
    #     "random_state": [1],
    #     "max_depth": [4, 5],
    #     "subsample": [0.8, 0.7]
    # }
    params = {
            "learning_rate": [0.1],
            "n_estimators": [700],
            "random_state": [1],
            "max_depth": [5],
            "subsample": [0.7]
        }
    boostingClassfier = GradientBoostingClassifier()

    grid = GridSearchCV(estimator=boostingClassfier, param_grid=params, cv=5, scoring="f1_micro")
    grid.fit(X, Y)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)


def train_ensemble(X_train, Y_train):
    # najbolji rezultat koji sam dobio: 0.7741935483870968 na test_preview
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8)
    clf.fit(X_train, Y_train)
    return clf


def show_boxplot(data, columns):
    for col in columns:
        plt.boxplot(data[col].to_numpy())
        plt.show()

def makeBoxPlotByRace(data):
    X_group1 = pd.concat([data[data['race'] == '1. White']]).to_numpy()
    X_group2 = pd.concat([data[data['race'] == "2. Black"]]).to_numpy()
    X_group3 = pd.concat([data[data['race'] == "3. Asian"]]).to_numpy()
    X_group4 = pd.concat([data[data['race'] == "4. Other"]]).to_numpy()

    #income data by continent
    X_group1_income = [x1[1] for x1 in X_group1]
    X_group2_income = [x2[1] for x2 in X_group2]
    X_group3_income = [x3[1] for x3 in X_group3]
    X_group4_income = [x4[1] for x4 in X_group4]

    plt.boxplot([X_group1_income, X_group2_income, X_group3_income, X_group4_income])
    plt.show()


    print("cnt White", len(X_group1_income))
    print("cnt Black", len(X_group2_income))
    print("cnt Asian", len(X_group3_income))
    print("cnt Other", len(X_group4_income))

def remove_outliers(data, columns):
    for col in columns:
        data_col = data[col]
        q1 = data_col.quantile(0.25)
        q3 = data_col.quantile(0.75)
        iqr = q3 - q1
        filter = (data_col < (q1 - 1.5 * iqr)) | (data_col > (q3 + 1.5 * iqr))
        outliers = data_col[filter]
        outliers_drop = outliers.dropna()
        for idx in outliers_drop.index:
            data.drop(idx, axis=0, inplace=True)


def statistic(data):
    X_group1 = pd.concat([data[data['race'] == '1. White']]).to_numpy()
    X_group2 = pd.concat([data[data['race'] == "2. Black"]]).to_numpy()
    X_group3 = pd.concat([data[data['race'] == "3. Asian"]]).to_numpy()
    X_group4 = pd.concat([data[data['race'] == "4. Other"]]).to_numpy()


    # ovo je za PROCENTE KOLIKO NJIH IMA KOJI STEPEN STUDIJA
    dict_white = {}
    dict_black = {}
    dict_asian = {}
    dict_other = {}

    for d in X_group1:
        if d[0] not in dict_white:
            dict_white[d[0]] = 0
        dict_white[d[0]] += 1

    print("for White")
    for key in dict_white.keys():
        print(key, dict_white[key]/len(X_group1))


    for d in X_group2:
        if d[0] not in dict_black:
            dict_black[d[0]] = 0
        dict_black[d[0]] += 1

    print("for Black")
    for key in dict_black.keys():
        print(key, dict_black[key]/len(X_group2))

    for d in X_group3:
        if d[0] not in dict_asian:
            dict_asian[d[0]] = 0
        dict_asian[d[0]] += 1

    print("for Asian")
    for key in dict_asian.keys():
        print(key, dict_asian[key] / len(X_group3))


    for d in X_group4:
        if d[0] not in dict_other:
            dict_other[d[0]] = 0
        dict_other[d[0]] += 1

    print("for Other")
    for key in dict_other.keys():
        print(key, dict_other[key] / len(X_group4))

def myLabelEncoder(data):
    for index, row in data.iterrows():
        retVal = 0
        if "Yes" in row['health_ins']:
            retVal = 1
        data.iloc[index, data.columns.get_loc('health_ins')] = retVal

# da se izracuna kovarijansa izmedju
def calculate_cov(data, pca_data):
    result = pd.concat([data, pca_data], axis=1, sort=False)
    corr = result.corr()
    # zaustaviti debagorom pa pogledati vrednosti
    plt.matshow(corr)
    plt.show()


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # print(train_data)
    # print(test_data)

    del train_data['year']   # msm da nema neki znacaj
    del test_data['year']

    train_data = train_data.dropna()
    train_data.reset_index(drop=True, inplace=True)
    # train_data = train_data.dropna(subset=['race'])
    # statistic(train_data)
    # statistic(test_data)

    # makeBoxPlotByRace(train_data)
    # makeBoxPlotByRace(test_data)
    # show_boxplot(train_data, ['year', 'age', 'wage'])
    # remove_outliers(train_data, ['year', 'age', 'wage'])
    #
    col_names = ['race','education', 'health', 'jobclass']
    # # col_names = ['jobclass', 'health', 'health_ins']
    for name in col_names:
        le = label_encoding(train_data, name)
        label_encoding(test_data, name, le)

    #radi samo za health_ins
    myLabelEncoder(train_data)
    myLabelEncoder(test_data)
    #
    # trebala bi i rasa ovde
    col_names = ['maritl']
    for name in col_names:
        train_data, ohe = one_hot_encoding(train_data, name)
        test_data = one_hot_encoding(test_data, name, ohe)
    #
    y_train = train_data['race'].to_numpy()
    del train_data['race']
    y_test = test_data['race'].to_numpy()
    del test_data['race']

    # col_names = ['year', 'age', 'wage']
    col_names = [ 'age', 'wage']
    for col in col_names:
        train_data[col], scaler = standard_scaler(train_data, col)
        test_data[col], _ = standard_scaler(test_data, col, scaler)

    # # PCA
    pca = PCA(svd_solver='full',n_components=5, copy=True)
    pca.fit(train_data)
    # print(pca.explained_variance_ratio_)
    # print(sum(pca.explained_variance_ratio_[0:7]))

    # sa 7 osa se zadrzi 97.5% varijanse


    #
    # # KernelPCA
    # # pca = KernelPCA(n_components=5)
    # # pca.fit(train_data)
    #

    # da se vidi sta PCA bira kako gleda.
    # train_data_pca = pd.DataFrame(data=pca.transform(train_data))
    # test_data_pca = pd.DataFrame(data=pca.transform(test_data))
    #
    # calculate_cov(train_data, train_data_pca)

    train_data = pd.DataFrame(data=pca.transform(train_data))
    test_data = pd.DataFrame(data=pca.transform(test_data))


    # # print(train_data)
    # # print(test_data)
    #
    x_train = train_data.to_numpy()
    #
    cross_validation(x_train, y_train)
    #
    # # model = SVC(gamma='scale', C=1)
    # # model = BaggingClassifier(n_estimators=10000)
    # # model = AdaBoostClassifier(n_estimators=100)
    # model = GradientBoostingClassifier(n_estimators=800, learning_rate=0.01, max_depth=4, subsample=0.7, random_state=1)
    # # # model = KNeighborsClassifier(n_neighbors=100)
    # #
    # model.fit(x_train, y_train)
    # #
    # x_test = test_data.to_numpy()
    # #
    # y_predict = model.predict(x_test)
    # #
    # score = calculate_f1_score(y_test, y_predict)
    # print(score)

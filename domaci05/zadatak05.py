import sys
import random

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import shapiro
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


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
        # print(x)
        # print(statistics.median(x))
        # print(statistics.mean(x))
        # print(min(x))
        # print(max(x))
        median.append(statistics.median(x))
        mean.append(statistics.mean(x))
        mini.append(min(x))
        maksi.append(max(x))

    median.append(statistics.median(all))
    mean.append(statistics.mean(all))
    mini.append(min(all))
    maksi.append(max(all))

    print('Asia', 'Africa', 'Americas', 'Europe', 'all')
    print("medians:", median)
    print("means:", mean)
    print("mins: ", mini)
    print("maxs:", maksi)

    # Analize oil Values
    oil_values = [val for val in data['oil']]
    has_oil = len([x for x in oil_values if x == "yes"])
    print("\n Analize oil values")
    print("has oil: ", has_oil, " hasn't oil: ", len(oil_values) - has_oil)

    print("\n")


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
    # regression_fill_nan(train)
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
        x_values = pd.concat([data[data['region'] == i[0]]['income'].dropna(),
                              data[data['region'] == i[1]]['income'].dropna()]).to_numpy()
        y_values = pd.concat([data[data['region'] == i[0]]['infant'].dropna(),
                              data[data['region'] == i[1]]['infant'].dropna()]).to_numpy()
        z_values = pd.concat(
            [data[data['region'] == i[0]]['oil'].dropna(), data[data['region'] == i[1]]['oil'].dropna()]).to_numpy()
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
        "init_params": ['kmeans', "random"]
    }

    gm = GaussianMixture()

    grid = GridSearchCV(estimator=gm, param_grid=params, cv=5, scoring="v_measure_score")
    grid.fit(X, Y)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)


def splitIntoGroups(data, group1=['Africa', 'Americas'], group2=['Europe', 'Asia']):

    X_group1 = pd.concat([data[data['region'] == group1[0]]])
                          # data[data['region'] == group1[1]]])
    X_group2 = pd.concat([data[data['region'] == group2[0]],
                          data[data['region'] == group2[1]],
                          data[data['region'] == group2[2]]
                          ])
    #
    # Y_group1 = pd.concat([data[data['region'] == group1[0]]['region'],
    #                       data[data['region'] == group1[1]]['region']]).to_numpy()
    #
    # Y_group2 = pd.concat([data[data['region'] == group2[0]]['region'],
    #                       data[data['region'] == group2[1]]['region']]).to_numpy()

    del X_group1['region']
    del X_group2['region']

    X_group1 = X_group1.to_numpy()
    Y_group1 = [0 for i in range(len(X_group1))]
    X_group2 = X_group2.to_numpy()
    Y_group2 = [1 for i in range(len(X_group2))]
    #samo po income da li ih je moguce razdvojiti kako treba
    # moze i plt da se vidi razlicitim bojama da li ih lepo razdvoji po ovom obelezju
    X_group1 = [x1[1] for x1 in X_group1]
    X_group2 = [x2[1] for x2 in X_group2]

    X_all = X_group1 + X_group2
    Y_all = Y_group1 + Y_group2

    X_all = np.asarray(X_all)
    Y_all = np.asarray(Y_all)

    X_all = X_all.reshape(-1, 1)
    # Y_all = Y_all.reshape(-1, 1)
    return X_all, Y_all


def gaussian_mixture(X_all_by_groups, Y_all_by_groups):
    # gm_group1 = GaussianMixture(n_components=2, max_iter=20000)
    # gm_group2 = GaussianMixture(n_components=2, max_iter=20000)

    cntTotal = len(Y_all_by_groups)
    cntG1 = len([y for y in Y_all_by_groups if y == 0])
    cntG2 = cntTotal - cntG1


    weights = [cntG1/cntTotal, cntG2/cntTotal]


    gm_all = GaussianMixture(n_components=2, max_iter=100000,
                             covariance_type='tied', n_init=10)
                             # ,weights_init=weights)
    # gm_group1.fit(X_train_group1, Y_train_group1)
    # gm_group2.fit(X_train_group2, Y_train_group2)
    gm_all.fit(X_all_by_groups)
    # gm_all.fit(X_all_by_groups, Y_all_by_groups)
     # return gm_group1, gm_group2, gm_all
    print(gm_all.weights_)
    return gm_all


def predict_gm(gm_group1, gm_group2, gm_all, X_test):
    Y_predict_all = gm_all.predict(X_test)
    print(Y_predict_all)
    return 'cao'


def makeBoxPlotByContinent(data):
    X_group1 = pd.concat([data[data['region'] == "Asia"]]).to_numpy()
    X_group2 = pd.concat([data[data['region'] == "Africa"]]).to_numpy()
    X_group3 = pd.concat([data[data['region'] == "Europe"]]).to_numpy()
    X_group4 = pd.concat([data[data['region'] == "Americas"]]).to_numpy()

    #income data by continent
    X_group1_income = [x1[0] for x1 in X_group1]
    X_group2_income = [x2[0] for x2 in X_group2]
    X_group3_income = [x3[0] for x3 in X_group3]
    X_group4_income = [x4[0] for x4 in X_group4]

    plt.boxplot([X_group1_income, X_group2_income, X_group3_income, X_group4_income])
    plt.show()


    # print("cnt Asia", len(X_group1_income))
    # print("cnt Africa", len(X_group2_income))
    # print("cnt Europe", len(X_group3_income))
    # print("cnt Americas", len(X_group4_income))


def removeOutLiersByIQRs(data):

    #remove Asian outliers
    asiaDf = data[data['region'] == "Asia"]
    asiaaQ1 = asiaDf.quantile(0.25)
    asiaaQ3 = asiaDf.quantile(0.75)
    asiaIQR = asiaaQ3 - asiaaQ1
    filterAsian = (asiaDf < (asiaaQ1 - 1.5 * asiaIQR)) | (asiaDf > (asiaaQ3 + 1.5 * asiaIQR))
    asianDf_outliers = asiaDf[filterAsian]
    asianDf_outliers_income = asianDf_outliers.dropna(subset=["income"])
    asianDf_outliers_infant = asianDf_outliers.dropna(subset=["infant"])

    for ind in asianDf_outliers_income.index:
        if ind in data.index:
            data.drop(ind, axis=0, inplace=True)
    for ind in asianDf_outliers_infant.index:
        if ind in data.index:
            data.drop(ind, axis=0, inplace=True)

    #remove Africa outliers
    africaDf = data[data['region'] == "Africa"]
    africaQ1 = africaDf.quantile(0.25)
    africaQ3 = africaDf.quantile(0.75)
    africaIQR = africaQ3 - africaQ1
    filterAfrica = (africaDf < (africaQ1 - 1.5 * africaIQR)) | (africaDf > (africaQ3 + 1.5 * africaIQR))
    africaDf_outliers = africaDf[filterAfrica]
    africaDf_outliers_income = africaDf_outliers.dropna(subset=["income"])
    africaDf_outliers_infant = africaDf_outliers.dropna(subset=["infant"])

    for ind in africaDf_outliers_income.index:
        if ind in data.index:
            data.drop(ind, axis=0, inplace=True)
    for ind in africaDf_outliers_infant.index:
        if ind in data.index:
            data.drop(ind, axis=0, inplace=True)

    #remove Americas outLiers
    americasDf = data[data['region'] == "Americas"]
    americasQ1 = americasDf.quantile(0.25)
    americasQ3 = americasDf.quantile(0.75)
    americasIQR = americasQ3 - americasQ1
    filterAmericas = (americasDf < (americasQ1 - 1.5 * americasIQR)) | (americasDf > (americasQ3 + 1.5 * americasIQR))
    americasDf_outliers = americasDf[filterAmericas]
    americasDf_outliers_income = americasDf_outliers.dropna(subset=["income"])
    americasDf_outliers_infant = americasDf_outliers.dropna(subset=["infant"])

    for ind in americasDf_outliers_income.index:
        if ind in data.index:
            data.drop(ind, axis=0, inplace=True)
    for ind in americasDf_outliers_infant.index:
        if ind in data.index:
            data.drop(ind, axis=0, inplace=True)

    data.reset_index(drop=True, inplace=True)


    #AsianDf je spreman i sad treba da ga vratimo u dataFrame nas..


    # print(asiaaQ1)
    # print(asiaaQ3)
    # print(asiaIQR)


#fora posto GMM koristi normalne distribucije da modeluje podatke za svaki feature, on pravi
# po par gausijana
# ae da proverimo za svaki feature po kontinetu da li podaci imaju normalnu raspodelu..
# ako nemaju da ih prvedemo u normalnu raspodelu...

def doStatisticTestForFeatureForContinet(data, region, feature):
    retVal = False

    if isinstance(data, pd.DataFrame):
        data_for_test = pd.concat([data[data['region'] == region][feature]]).to_numpy()
    else:
        data_for_test = data

    stat, p = shapiro(data_for_test)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('For region', region, " feature: ", feature.upper(), ' looks Gaussian (fail to reject H0)')
        retVal = True
    else:
        print('For region', region, " feature: ", feature.upper(), ' does NOT look Gaussian (reject H0)')
    return retVal


def mapFeatureToNormalDistByQuantile(data, region_features):
    # quantile_transformer = preprocessing.\
    #     QuantileTransformer(output_distribution='normal', random_state=0)

    # odma uradi i MAPIRANJE NA NORMALNU I NORMALIZACIJu..
    transformer = preprocessing.PowerTransformer(method='box-cox', standardize=True)

    for region_feature in region_features:
        region = region_feature[0]
        feature = region_feature[1]
        data_for_normalisation = pd.concat([data[data['region'] == region][feature]])

        print("PRE NORMALIZACIJE")
        isNormalDist = doStatisticTestForFeatureForContinet(data_for_normalisation.to_numpy(), region, feature)
        X_trans = []

        if not isNormalDist:
            X_trans = transformer.fit_transform(data_for_normalisation.to_numpy().reshape(-1, 1))
            print("POSLE NORMALIZACIJE")
            doStatisticTestForFeatureForContinet(X_trans, region, feature)

        #ova sto su vec normalna po default njih samo sklairaj po Z score-u
        else:
            scaler = StandardScaler()
            X_trans = scaler.fit_transform(data_for_normalisation.to_numpy().reshape(-1, 1))

        # na odogovarajuce indekse da se podese odgovarajuce VREDNOSTI
        for indxArray, trans_x in enumerate(data_for_normalisation.axes[0]):
            indxDatFrame = trans_x
            data.loc[indxDatFrame,feature] = X_trans[indxArray]


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
            row['infant'] = reg.predict(np.array([[row['oil'], row['region']]]))[0]


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    data_preprocessing(train_data, test_data)
    # statistics_infant(train_data)
    # show_3D_plot(train_data)
    # makeBoxPlotByContinent(train_data)
    removeOutLiersByIQRs(train_data)
    # makeBoxPlotByContinent(train_data)
    # statistics_infant(train_data)


    # KONTAM DA OVDE IMA MEST ZA NAPREDAK modela pre bilo cega drugog.
    doStatisticTestForFeatureForContinet(train_data, "Americas", "income")
    # Africa, infant ----> IMA NORMANU RASPODELU
    # Africa, income ----> NEMA
    # EU, infant     ----> NEMA
    # EU, income     ----> IMA
    # Asia, infant   ----> IMA
    # Asia, income   ----> NEMA
    # America, infant ---> IMA
    # America, income ---> IMA

    # DOBIJU SE LOSIJI REZULTATI, ali moze sve da ih namapira na NORMALNU...
    # mada realno NE ZNAS TI KOJEM KLASTERU KOJE PRIPADA PA DA RADIS ovo, da znas stace ti onda klasterizacija..


    # all_combinations = [["Africa", "income"], ["Africa", "infant"],
    #                     ["Europe", "income"], ["Europe", "infant"],
    #                     ["Asia", "income"], ["Asia", "infant"],
    #                     ["Americas", "income"], ["Americas", "infant"]]
    # mapFeatureToNormalDistByQuantile(train_data, all_combinations)

    # #
    # X_all_train, Y_all_train = splitIntoGroups(train_data, group1=["Africa"], group2=["Europe", "Asia", "Americas"])
    # #
    # #
    # #
    # gm = gaussian_mixture(X_all_train, Y_all_train)
    # Y_predict_train = gm.predict(X_all_train)
    # score = calculate_v_measure_score(Y_all_train, Y_predict_train)
    # print("train score only by income", score)
    # #
    # # misClusteredByIncome = len([1 for indx, x_v in enumerate(Y_predict_train) if Y_all_train[indx] != x_v])
    # # print("misClustered by income on train dataSet cnt: ", misClusteredByIncome)
    # # #
    # X_test, Y_test = splitIntoGroups(test_data, group1=["Africa"], group2=["Europe", "Asia", "Americas"])
    # Y_predict_test = gm.predict(X_test)
    # score = calculate_v_measure_score(Y_test, Y_predict_test)
    # print("test score only by income", score)



    # Y_train = train_data['region'].to_numpy()
    # del train_data['region']
    # X_train = train_data.to_numpy()
    # #
    # X_train, scaler = minMaxScaler(X_train, None)
    #
    # cross_validation(X_train, Y_train)
    # #
    # gm = GaussianMixture(n_components=4, covariance_type='diag', max_iter=100000, n_init=100)
    # gm.fit(X_train)
    # Y_predict_train = gm.predict(X_train)
    # score = calculate_v_measure_score(Y_train, Y_predict_train)
    # print("trainScore: ", score)
    #
    # Y_test = test_data['region'].to_numpy()
    # del test_data['region']
    # X_test = test_data.to_numpy()
    # X_test, scaler = minMaxScaler(X_test, scaler)
    # #
    # Y_predict = gm.predict(X_test)
    # score = calculate_v_measure_score(Y_test, Y_predict)
    # print("testScore: ", score)

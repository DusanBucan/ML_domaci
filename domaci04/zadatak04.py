import sys
import random
# from datetime import datetime

import numpy as np
import pandas as pd
# from sklearn import linear_model

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, \
    GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


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
    pass
    # cnt_per_class = {}
    # for tr_instance in tr_data.values:
    #     speed_class = tr_instance[0]
    #     if speed_class in cnt_per_class.keys():
    #         cnt_per_class[speed_class] += 1
    #     else:
    #         cnt_per_class[speed_class] = 1
    #
    # objects = cnt_per_class.keys()
    # y_pos = np.arange(len(objects))
    # performance = [cnt_per_class[ob] for ob in objects]
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.show()

#subsample 0.5
    # probati onako na orginalnom skupu
    # probati na ovim sto si ih ti pravi da budu izbalansirani

# Boosting metode --> sa slabijim modelima, stump smo gurali
    # AdaBoost
        # 0.42 ovako kad podelim dobije se na validacionom...

    # GradientBoost

# Bagging metode --> trebal bi da rade sa jacim modelima

    # dat primer sa SVC-om..
        # linearni kernel sa 10 estimatora, C = 1, max_samples = 0.8 bilo oko 0.4877

    # priemer sa DecisionTreeClassifier oko 0.52, depth=3, 20 estimatora i 0.8 od celog skupa


# Glasanja

def crossValidation(X, Y):

    # a1 = BernoulliNB()
    # a2 = ExtraTreesClassifier(n_estimators=200)  # spor
    # a3 = linear_model.RidgeClassifier(alpha=0.001)  # bitno
    # a4 = KNeighborsClassifier(n_neighbors=55, weights='distance', algorithm='ball_tree', leaf_size=8)
    # a5 = SVC(C=2)
    # params = {}
    # svi 0.5348426283821094
    # bez a2 0.5347874102705688
    # BernoulliNB 0.5192711209276643
    # ExtraTreesClassifier ispod 0.5
    # RidgeClassifier 0.5273881833241303
    # knn 0.5186637217007178

    # grid = GridSearchCV(estimator=VotingClassifier([('a1', a1), ('a3', a3), ('a4', a4), ('a5', a5)]), param_grid=params,
    #                     cv=5)

    # grid = grid.fit(X, Y)
    # print(grid.best_estimator_)
    # print(grid.best_score_)
    # print(grid.best_params_)


    # GradientBoostingClassifier 0.53837 0.2
    # params = {"n_estimators": [100], "learning_rate": [0.1, 0.2]}
    # grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=params, cv=5)


    # grid.fit(X, Y)
    # print("zavrsio grid")
    # print(grid.best_estimator_)
    # print(grid.best_score_)
    # print(grid.best_params_)

    # params = {"base_estimator__C": [0.5, 1, 1.5, 2],
    #           "base_estimator__dual": [False],
    #           "n_estimators": [10, 20],
    #            "max_samples": [0.5, 0.8]
    #         }

    # baggingClassfier = BaggingClassifier(base_estimator=SVC())
    # baggingClassfier = BaggingClassifier(base_estimator=LinearSVC())


    params = {
        "n_estimators": [10, 20,30,40, 60],
        "max_samples": [0.7, 0.8, 1],
        "bootstrap": [True],
        "base_estimator__max_depth": [4, 5, 6, 7, 8],
        "base_estimator__random_state": [0, 1, 2, 3, 4],
    }
    # baggingClassfier = BaggingClassifier(base_estimator=DecisionTreeClassifier())

    params = {
        "learning_rate": [0.1],
        "n_estimators": [700],
        "random_state": [1],
        "max_depth": [5, 6],
        "subsample": [0.8]
    }
    boostingClassfier = GradientBoostingClassifier()

    grid = GridSearchCV(estimator=boostingClassfier, param_grid=params, cv=5, scoring="f1_micro")
    grid.fit(X, Y)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)



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




def upsample(data_class, Y, desired_instances_cnt):
    resampled_data = resample(data_class, n_samples=desired_instances_cnt, random_state=1)
    y = [Y for i in range(0, len(resampled_data))]
    return resampled_data, y


def makeBalancedDataSet(Y_train, X_train):
    # ima ih 483..
    X_train_1_9 = [x for y, x in zip(Y_train, X_train) if y == "1-9km/h"]
    X_train_1_9_resampled, y_1_9_resampled = upsample(X_train_1_9, "1-9km/h", 4000)

    # ima ih 1024 orginalno
    X_train_55 = [x for y, x in zip(Y_train, X_train) if y == "55+"]
    X_train_55_resampled, y_55_resampled = upsample(X_train_55, "55+", 4000)

    # ima ih 2041 orginalno
    X_train_40_54 = [x for y, x in zip(Y_train, X_train) if y == "40-54"]
    X_train_40_54_resampled, y_40_54_resampled = upsample(X_train_40_54, "40-54", 4000)

    # ima ih 5671 orginalno
    X_train_25_39 = [x for y, x in zip(Y_train, X_train) if y == "25-39"]
    X_train_25_39_resampled, y_25_39_resampled = upsample(X_train_25_39, "25-39", 8000)

    X_train_10_24 = [x for y, x in zip(Y_train, X_train) if y == "10-24"]
    X_train_10_24_resampled, y_10_24_resampled = upsample(X_train_10_24, "10-24", 8000)

    X_train1 = []
    Y_train1 = []

    X_train2 = []
    Y_train2 = []

    X_train1 += X_train_1_9_resampled
    Y_train1 += y_1_9_resampled

    X_train2 += X_train_1_9_resampled
    Y_train2 += y_1_9_resampled


    X_train1 += X_train_10_24_resampled[0:4000]
    Y_train1 += y_10_24_resampled[0:4000]

    X_train2 += X_train_10_24_resampled[4000: len(X_train_10_24_resampled)]
    Y_train2 += y_10_24_resampled[4000:len(y_10_24_resampled)]


    X_train1 += X_train_25_39_resampled[0:4000]
    Y_train1 += y_25_39_resampled[0:4000]

    X_train2 += X_train_25_39_resampled[4000: len(X_train_25_39_resampled)]
    Y_train2 += y_25_39_resampled[4000:len(y_25_39_resampled)]


    X_train1 += X_train_40_54_resampled
    Y_train1 += y_40_54_resampled

    X_train2 += X_train_40_54_resampled
    Y_train2 += y_40_54_resampled


    X_train1 += X_train_55_resampled
    Y_train1 += y_55_resampled

    X_train2 += X_train_55_resampled
    Y_train2 += y_55_resampled

    train1 = [np.concatenate((x, [y])) for x, y in zip(X_train1, Y_train1)]
    random.shuffle(train1)
    X_train1 = [xy[0:len(xy)-1] for xy in train1]
    Y_train1 = [xy[len(xy)-1] for xy in train1]

    train2 = [np.concatenate( (x, [y])) for x, y in zip(X_train2, Y_train2)]
    random.shuffle(train2)
    X_train2 = [xy[0:len(xy) - 1] for xy in train2]
    Y_train2 = [xy[len(xy)-1] for xy in train2]


    return X_train1, Y_train1, X_train2, Y_train2


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # print(datetime.now())

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Uklanjanje NaN vrednosti iz trening skupa
    train_data = train_data.dropna()

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # statistic(train_data)


    Y_train = train_data['speed'].to_numpy()
    del train_data['speed']
    X_train = train_data.to_numpy()

    # X_train1, Y_train1, X_train2, Y_train2 = makeBalancedDataSet(Y_train, X_train)

    # sklairana koriscenjem Normalne Raspodele
    # X_train1, scaler = scale_data(X_train1)
    # X_train, scaler = scale_data(X_train)

    #izdvajanje obelezja uz pomoc Stabla
    # X_train, selector = featureSelectionTreBased(X_train, Y_train, None)

    # crossValidation(X_train, Y_train)
    #
    # clf = SVC(C=2)
    # clf = AdaBoostClassifier(learning_rate=0.1, n_estimators=900)
    # clf = RandomForestClassifier(n_estimators=10)

    # 0.5234 se dobije...
    # clf = baggingClassfier = BaggingClassifier(
    #         base_estimator=LinearSVC(C=0.5, class_weight="balanced", dual=False),
    #         n_estimators=10,
    #         max_samples=0.5)

    # clf = BaggingClassifier(
    #         base_estimator=DecisionTreeClassifier(max_depth=6, random_state=1),
    #         n_estimators=10,
    #         max_samples=0.8,
    #         bootstrap=True)

    # params = {
    #     "learning_rate": [0.1, 0.2],
    #     "n_estimators": [300],
    #     "random_state": [1],
    #     "max_depth": [4, 5, 6],
    #     "subsample": [0.8, 1]
    # }
    #daje 58.3 na testPreview sa 300 estimatora i dubina 4 subsample =0.8
    #daje 61,7 na testPreview sa 300 estimatora i dubina 5 subsample =0.8
    #daje 62,4 na testPreview sa 300 estimatora i dubina 6 subsample =0.8

    # 0.5838926174496645 na TestPreview sa 300 estimatora i dubina 7 subsample =0.8 BIO NAJBOLJI NA cross
    # 0.59 na TestPreview BEZ NORMALIZACIJE sa 300 estimatora i dubina 7 subsample =0.8 BIO NAJBOLJI NA cross
        #==> ponovio 2 puta za redom nekako mi je najrealniji...

    #daje 64 na testPreview sa 700 estimatora i dubina 5 subsample =0.8

        # 0.6174496644295302 kad se ne normalizuju
        # 0.6442953020134228 kad se normalizuju

    #700, 6 se ne uklapa u vreme od 180 sek...


    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300,
                                     random_state=1, max_depth=7, subsample=0.8)





    clf = clf.fit(X_train, Y_train)
    # # #
    Y_test = test_data['speed'].to_numpy()
    del test_data['speed']
    X_test = test_data.to_numpy()

    # X_test, scaler = scale_data(X_test, scaler)
    # X_test, selector = featureSelectionTreBased(X_test, None, selector)
    #
    Y_predict = clf.predict(X_test)
    score = calculate_micro_f1_score(Y_test, Y_predict)
    print(score)

    # print(datetime.now())
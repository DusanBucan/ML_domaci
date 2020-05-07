import sys
import json
import string
from math import ceil

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# reci koje su u stopwords a click su
clic_stopwords = ["we", "you", "your", "yourself", "they", "them", "their", "theirs", "themselves", "what", "which",
                  "who", "this", "that", "these", "why", "how", "all"]

class Article:
    def __init__(self, clickbait, text):
        self.clickbait = clickbait
        self.text = text
        self.processedText = None
        self.words = []


def load_from_file(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)

        articles = []
        for d in data:
            articles.append(Article(d['clickbait'], d['text']))

        return articles


def preprocess_text(data):
    for article in data:
        # prebaci sve na mala slova
        article.text = article.text.lower()

        # obrisi znakove interpunkcije
        article.text = article.text.translate(str.maketrans('', '', string.punctuation))

        # izbaci stopwords i brojeve
        article.words = article.text.split(' ')
        article.processedText = " ".join([word for word in article.words if not (word in stopwords) or word in clic_stopwords])
        # article.words = article.processedText.split(' ')
        # print(article.words)


def initBoW(data):
    corpus = [article.processedText for article in data]
    # v = CountVectorizer()
    v = CountVectorizer(ngram_range=(1, 2), binary=False)
    # v = CountVectorizer(ngram_range=(1, 2), binary=True)
    v.fit(corpus)
    return v


def initTfiDf(data):
    corpus = [article.processedText for article in data]
    v = TfidfVectorizer()
    v.fit(corpus)
    return v


def initHashVectorization(n_features=2**16):
    return HashingVectorizer(n_features=n_features)


def vectorize(data, v):
    corpus = [article.processedText for article in data]
    vectors = v.transform(corpus)
    return vectors


def calculate_F1_score(Y_true, Y_predicted):
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_predicted).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision: ", precision)
    print("recall: ", recall)
    return (2 * precision * recall) / (precision + recall)


def train_model(model, data, v):
    vectors = vectorize(data, v)
    X = [vector.toarray()[0] for vector in vectors]
    Y = [article.clickbait for article in data]
    model.fit(X, Y)
    print(" ===== gotov trening =====")


def predict(model, data, v):
    vectors = vectorize(data, v)
    X = [vector.toarray()[0] for vector in vectors]
    Y_true = [article.clickbait for article in data]
    Y_predicted = model.predict(X)

    good = 0
    wrongClassified = []
    for indx, trueValue in enumerate(Y_true):
        if trueValue == Y_predicted[indx]:
            good += 1
        else:
            wrongClassified.append(data[indx].text)

    f1Score = calculate_F1_score(Y_true, Y_predicted)
    # print("score: ",  (good / len(Y_true)), "\n")
    print("F1 score: ", f1Score, "\n")
    print("pogresno klasifikovao tekstove", len(wrongClassified))
    #print(wrongClassified)

    return f1Score, (good / len(Y_true))


def statistic(data):
    total = len(data)
    clbNo = 0
    normalnNo = 0
    for article in data:
        if article.clickbait:
            clbNo += 1
        else:
            normalnNo += 1

    print("total: ", total)
    print("clickbait: ", clbNo / total)
    print("normal: ", normalnNo / total)


# svaka grupa da ima dobar odnos..
# podelis dobre na 5, podelis normalne na 5 i onda spajas
def stratification(clickBaitArticles, notClickBaitArticles, k_fold=10):
    folds = {}

    numClickBaitArticlePerFold = ceil(len(clickBaitArticles) / k_fold)
    numNotClickBaitArticlesPerFold = ceil(len(notClickBaitArticles) / k_fold)

    j = 0
    for i in range(0, len(clickBaitArticles), numClickBaitArticlePerFold):
        if j not in folds.keys():
            folds[j] = []
        a = clickBaitArticles[i: i + numClickBaitArticlePerFold]
        folds[j].append(a)
        j += 1

    j = 0
    for i in range(0, len(notClickBaitArticles), numNotClickBaitArticlesPerFold):
        if j not in folds.keys():
            folds[j] = []
        folds[j].append(notClickBaitArticles[i: i + numNotClickBaitArticlesPerFold])
        j += 1

    for key in folds.keys():
        folds[key] = folds[key][0] + folds[key][1]
    return folds


def cross_validation(stratified_data):
    accuracy = []
    f1_score = []
    accuracy_tr = []
    f1_score_tr = []
    for key in stratified_data.keys():
        tr_data = []
        valid_data = stratified_data[key]
        svm = LinearSVC()

        for key2 in stratified_data.keys():
            if key2 != key:
                tr_data += stratified_data[key2]

        # skupovi su podeseni sad istreniras i evaluiras na validacionom
        # TODO: odulucti se za jedan od ova 3
        # v = initHashVectorization()
        v = initTfiDf(tr_data)
        # v = initBoW(tr_data)
        train_model(svm, tr_data, v)
        f1, a = predict(svm, valid_data, v)
        f1_score.append(f1)
        accuracy.append(a)
        f1, a = predict(svm, tr_data, v)
        f1_score_tr.append(f1)
        accuracy_tr.append(a)

    print("valid")
    print("F1")
    # print(f1_score)
    print(sum(f1_score)/len(f1_score))
    print("accuracy")
    # print(accuracy)
    print(sum(accuracy)/len(accuracy))
    print("train")
    print("F1")
    # print(f1_score_tr)
    print(sum(f1_score_tr) / len(f1_score_tr))
    print("accuracy")
    # print(accuracy_tr)
    print(sum(accuracy_tr) / len(accuracy_tr))


def statistic_words(data):
    non_dict = {}
    clic_dict = {}
    all_dict = {}
    stopwords_dict = dict((w, 0) for w in stopwords)
    stopwords_clic_dict = dict((w, 0) for w in stopwords)
    stopwords_non_dict = dict((w, 0) for w in stopwords)
    for a in data:
        for word in a.words:
            if word not in all_dict.keys():
                all_dict[word] = 0
            all_dict[word] += 1
            if a.clickbait:
                if word not in clic_dict:
                    clic_dict[word] = 0
                clic_dict[word] += 1
            else:
                if word not in non_dict:
                    non_dict[word] = 0
                non_dict[word] += 1
            if word in stopwords_dict:
                stopwords_dict[word] += 1
            if word in stopwords_clic_dict and a.clickbait:
                stopwords_clic_dict[word] += 1
            if word in stopwords_non_dict and not a.clickbait:
                stopwords_non_dict[word] += 1

    print({k: v for k, v in sorted(all_dict.items(), key=lambda item: item[1], reverse=True)})
    print({k: v for k, v in sorted(clic_dict.items(), key=lambda item: item[1], reverse=True)})
    print({k: v for k, v in sorted(non_dict.items(), key=lambda item: item[1], reverse=True)})
    print({k: v for k, v in sorted(stopwords_dict.items(), key=lambda item: item[1], reverse=True)})
    print({k: v for k, v in sorted(stopwords_clic_dict.items(), key=lambda item: item[1], reverse=True)})
    print({k: v for k, v in sorted(stopwords_non_dict.items(), key=lambda item: item[1], reverse=True)})

    # for key in stopwords_clic_dict:
    #     print(key, stopwords_clic_dict[key], stopwords_non_dict[key])

    clic_dict = {k: v for k, v in sorted(clic_dict.items(), key=lambda item: item[1], reverse=True)}
    non_dict = {k: v for k, v in sorted(non_dict.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for key in clic_dict:
        if key in non_dict:
            print(key, clic_dict[key], non_dict[key])
        else:
            print(key, clic_dict[key], 0)
        i += 1
        if i == 20:
            break
    print("---------------------")
    i = 0
    for key in non_dict:
        if key in clic_dict:
            print(key, clic_dict[key], non_dict[key])
        else:
            print(key, 0, non_dict[key])
        i += 1
        if i == 20:
            break



def split_articles_by_class(data):
    clickBaitArticles = [article for article in data if article.clickbait]
    notClickBaitArticles = [article for article in data if not article.clickbait]
    return clickBaitArticles, notClickBaitArticles

# koliko ces da uzmes iz svakog fold-a
def plot_learning_curve(data):

    training_scores = []
    test_scores = []
    data_size = [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
    clickBaitArticles, notClickBaitArticles = split_articles_by_class(data)


    for size in data_size:
        cbArtl = clickBaitArticles[0:len(clickBaitArticles)*size]
        notCbAtrl = notClickBaitArticles[0:len(notClickBaitArticles)]
        folds_ = stratification(cbArtl, notCbAtrl)
        scores_for_size = cross_validation(folds_)
        training_scores.append(scores_for_size["avg_tr_score"])
        test_scores.append(scores_for_size["avg_valid_score"])


    # TODO: da se plotuje grafik




if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = load_from_file(train_path)
    test_data = load_from_file(test_path)

    preprocess_text(train_data)
    preprocess_text(test_data)

    # statistic_words(train_data)
    CBArticles, notCBArticles = split_articles_by_class(train_data)
    folds = stratification(CBArticles, notCBArticles)
    cross_validation(folds)

    # svm = LinearSVC()
    # vectorizer = initBoW(train_data)
    # train_model(svm, train_data, vectorizer)
    # predict(svm, test_data, vectorizer)

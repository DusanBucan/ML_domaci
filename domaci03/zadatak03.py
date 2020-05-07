import sys
import json
import string
from math import ceil

from sklearn.feature_extraction.text import CountVectorizer
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
        words = article.text.split(' ')
        article.processedText = " ".join([word for word in words if not ((word in stopwords) or word.isdigit())])

        # print(article.words)

def initBoW(data):
    corpus = [article.processedText for article in data]
    # v = CountVectorizer()
    v = CountVectorizer(ngram_range=(1, 2))
    v.fit(corpus)
    return v


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

    print("pogresno klasifikovao tekstove")
    print(wrongClassified)

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
    print("clickbait: ", clbNo/total)
    print("normal: ", normalnNo/total)

#svaka grupa da ima dobar odnos..
#podelis dobre na 5, podelis normalne na 5 i onda spajas
def stratification(data, k_fold=5):

    folds = {}

    clickBaitArticles = [article for article in data if article.clickbait]
    notClickBaitArticles = [article for article in data if not article.clickbait]

    numClickBaitArticlePerFold = ceil(len(clickBaitArticles) / k_fold)
    numNotClickBaitArticlesPerFold = ceil(len(notClickBaitArticles) / k_fold)

    j = 0
    for i in range(0, len(clickBaitArticles), numClickBaitArticlePerFold):
        if j not in folds.keys():
            folds[j] = []
        a = clickBaitArticles[i: i+numClickBaitArticlePerFold]
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

    for key in stratified_data.keys():
        tr_data = []
        valid_data = folds[key]
        svm = LinearSVC()

        for key2 in stratified_data.keys():
            if key2 != key:
                for article in stratified_data[key2]:
                    tr_data.append(article)

        #skupovi su podeseni sad istreniras i evaluiras na validacionom
        v = initBoW(tr_data)
        train_model(svm, tr_data, v)
        predict(svm, valid_data, v)

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = load_from_file(train_path)
    test_data = load_from_file(test_path)

    preprocess_text(train_data)
    preprocess_text(test_data)

    folds = stratification(train_data)
    cross_validation(folds)

    # svm = LinearSVC()
    # vectorizer = initBoW(train_data)
    # train_model(svm, train_data, vectorizer)
    # predict(svm, test_data, vectorizer)

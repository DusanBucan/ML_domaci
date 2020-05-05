import sys
import json

class Article:
    def __init__(self, clickbait, text):
        self.clickbait = clickbait
        self.text = text

def load_from_file(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)

        articles = []
        for d in data:
            articles.append(Article(d['clickbait'], d['text']))

        return data

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = load_from_file(train_path)
    test_data = load_from_file(test_path)

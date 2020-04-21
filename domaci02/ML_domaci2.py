import matplotlib.pyplot as plt
import pandas as pd

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


if __name__ == '__main__':
    trainPath = 'dataset/train.csv'
    testPath = 'dataset/test_preview.csv'

    train_data = pd.read_csv(trainPath)
    test_data = pd.read_csv(testPath)

    view_data(train_data, test_data)

import pandas as pd

from sklearn.linear_model import (RANSACRegressor, HuberRegressor)
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')
    print(dataset.head(5))

    X = dataset.drop(['country', 'score'], axis=1)
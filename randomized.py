import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.head())
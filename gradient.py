import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())
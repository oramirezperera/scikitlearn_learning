import pandas as pd
import sklearn

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')

    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    dt_features = StandardScaler().fit_transform(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target)
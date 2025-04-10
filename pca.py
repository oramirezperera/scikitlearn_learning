# Importing general libraries
import pandas as pd
import sklearn

# Importing specific modules
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Charging the data to a pandas dataframe
    dt_heart = pd.read_csv('./data/heart.csv')

    # Print the head to see if charged right
    print(dt_heart.head(5))

    # Dropping the column target
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # Normalizing data
    dt_features = StandardScaler().fit_transform(dt_features)

    # Splitting the data to training and test, added random state to get fixed results
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    # n_components = min(n_samples, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)

    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    dt_train2 = ipca.transform(X_train)
    dt_test2 = ipca.transform(X_test)

    logistic.fit(dt_train2, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test2, y_test))

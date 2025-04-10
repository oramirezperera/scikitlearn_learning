# Importing general libraries
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Importing specific modules
from sklearn.decomposition import KernelPCA

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

    # Here we define which type of kernel we will use
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)
    print("SCORE KPCA: ", logistic.score(dt_test, y_test))

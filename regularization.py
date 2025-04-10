# General libraries
import pandas as pd
import sklearn

# Specific modules
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# Split test and MSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    modelenet = ElasticNet(random_state=0, alpha=0.2).fit(X_train,y_train)
    y_predict_enet = modelenet.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss: ", ridge_loss)

    enet_loss = mean_squared_error(y_test, y_predict_enet)
    print("Enet loss: ", enet_loss)

    print("=" * 32)
    print("Coef Lasso")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef Ridge")
    print(modelRidge.coef_)

    print("="*32)
    print("Coef enet")
    print(modelenet.coef_)
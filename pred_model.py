import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def predict_price(arr):
    # reading the csv file to train the model

    dframe = pd.read_csv('datasets/train.csv')

    # getting features and labels from the data

    x, y = dframe[['bhk', 'area', 'ready_to_move', 'longitude', 'latitude']], dframe['price']

    # splititng the testing and training data

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

    # training our model

    DtRgr = DecisionTreeRegressor(random_state=0)

    DtRgr.fit(x_train, y_train)

    # Now testing our model by passing the test data in the model that has made

    y_pred = DtRgr.predict(x_test)

    # Now calculating the coefficient of determination

    rsq = r2_score(y_test, y_pred)

    # Now calculating the mean squared error

    mse = mean_squared_error(y_test, y_pred)

    # Now calculating the price of the data what we got in the method

    pred_price = DtRgr.predict(arr)

    dict = {
        'price': int(float(pred_price)),
        'accuracy': rsq*100,
        'mean squared error': mse,
        'model': 'DecisionTree Regressor'
    }

    return dict


if __name__ == '__main__':
    my_features = [[2, 896.7741935, 1, 26.832353, 75.841749]]
    tt = predict_price(my_features)
    price = tt['price']

    print(type(int(float(price))))
    print(f"The price of your house is: {price - 1} Lakhs")

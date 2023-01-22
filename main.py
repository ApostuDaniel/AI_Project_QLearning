import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import parse
import utils


if __name__ == '__main__':

    all_data = pd.read_csv(r'resources/iris.data', header=0, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    all_data['class'] = all_data['class'].apply(lambda name: parse.iris_name_to_number(name))
    target = ['class']
    variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = all_data[variables].values
    y = all_data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=4))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # build the model
    model.fit(X_train, y_train, epochs=50, verbose=0)

    pred_train = model.predict(X_train)
    print(pred_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(X_test)
    print(pred_test)
    scores2 = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

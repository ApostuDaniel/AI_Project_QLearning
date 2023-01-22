import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import parse
import utils
from DeepQ_Agent import Agent
from IrisGym import IrisGym


def reinforcement_learning(variables_train, target_train, variables_test, target_test):
    env = IrisGym(dataset=(variables_train, target_train))
    lr = 0.001
    n_games = 1500
    agent = Agent(gamma=0.01, epsilon=1.0, lr=lr,
                  input_dims=env.observation_space.shape,
                  n_actions=3, mem_size=100000, batch_size=1,
                  epsilon_end=0.01)
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            done = done
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score,
              'average_score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    # filename = 'lunarlander_tf2.png'
    # x = [i + 1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)


def supervized_learning(variables_train, target_train, variables_test, target_test):
    model = Sequential([
        Dense(50, activation='relu', input_dim=4),
        Dense(40, activation='relu'),
        Dense(3, activation='softmax')])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # build the model
    model.fit(variables_train, target_train, epochs=50, verbose=0)

    pred_train = model.predict(variables_train)
    scores = model.evaluate(variables_train, target_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(variables_test)
    scores2 = model.evaluate(variables_test, target_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))


if __name__ == '__main__':
    all_data = pd.read_csv(r'resources/iris.data', header=0,
                           names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    all_data['class'] = all_data['class'].apply(lambda name: parse.iris_name_to_number(name))
    target = ['class']
    variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = all_data[variables].values
    y = all_data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    #pentru reinforcement learning decomentati linia asta
    reinforcement_learning(X_train, y_train, X_test, y_test)

    # pentru learning cu reteaua neuronala decomentatie urmatoarele linii
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # supervized_learning(X_train, y_train, X_test, y_test)

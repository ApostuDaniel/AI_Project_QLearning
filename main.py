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

from matplotlib import pyplot as plt

from matplotlib import pyplot as plt

def reinforcement_learning(variables_train, target_train, variables_test, target_test, episodes, network_weights=None):
    env = IrisGym(dataset=(variables_train, target_train), images_per_episode=len(variables_train))
    lr = 0.001
    n_games = episodes
    agent = Agent(gamma=0.01, epsilon=1.0, lr=lr,
                  input_dims=env.observation_space.shape,
                  n_actions=3, mem_size=100000, batch_size=1,
                  epsilon_end=0.01)
    if network_weights is not None:
        #to initialize the weights we need to call predict once
        agent.q_eval.predict(np.asarray(variables_train[0]).reshape((1, len(variables_train[0]))), verbose=0)
        agent.q_eval.set_weights(network_weights)
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
        accuracy
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score,
              'average_score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    if show_evolution:
        pass
    if show_boxplot:
        pass

    return agent.q_eval
    # filename = 'lunarlander_tf2.png'
    # x = [i + 1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)


def supervized_learning(variables_train, target_train, variables_test, target_test, epochs, network_weights=None, show_boxplot=False, show_evolution=True):
    model = Sequential([
        Dense(50, activation='relu', input_dim=4),
        Dense(40, activation='relu'),
        Dense(3, activation='softmax')])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    if network_weights is not None:
        model.predict(np.asarray(variables_train[0]).reshape((1, len(variables_train[0]))), verbose=0)
        model.set_weights(network_weights)

    # build the model
    history = model.fit(variables_train, target_train, epochs=epochs, verbose=1)

    pred_train = model.predict(variables_train)
    scores = model.evaluate(variables_train, target_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(variables_test)
    scores2 = model.evaluate(variables_test, target_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

    if show_boxplot:
        plt.boxplot([history.history['accuracy'], history.history['loss']], labels=['accuracy', 'loss'])
        plt.title('Supervized learning boxplot')
        plt.show()
    if show_evolution:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['loss'])
        plt.legend(['accuracy', 'loss'], loc='upper left')
        plt.title('Supervized learning evolution')
        plt.show()
    return model


def supervized_to_RL(variables_train, target_train, variables_test, target_test, epochs, episodes, show_boxplot=False):
    y_train = to_categorical(target_train)
    y_test = to_categorical(target_test)
    trained_network = supervized_learning(variables_train, y_train, variables_test, y_test, epochs)
    reinforcement_learning(variables_train, target_train, variables_test, target_test, episodes,
                           trained_network.get_weights())


def RL_to_supervized(variables_train, target_train, variables_test, target_test, epochs, episodes, show_boxplot=False):
    y_train = to_categorical(target_train)
    y_test = to_categorical(target_test)
    trained_network = reinforcement_learning(variables_train, target_train, variables_test, target_test, episodes, None)
    supervized_model = supervized_learning(variables_train, y_train, variables_test, y_test, epochs,
                                           trained_network.get_weights())


if __name__ == '__main__':
    all_data = pd.read_csv(r'resources/iris.data', header=0,
                           names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    all_data['class'] = all_data['class'].apply(lambda name: parse.iris_name_to_number(name))
    target = ['class']
    variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = all_data[variables].values
    y = all_data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # supervized_to_RL(X_train, y_train, X_test, y_test, 50, 10)
    RL_to_supervized(X_train, y_train, X_test, y_test, 50, 10)

    # pentru reinforcement learning decomentati linia asta
    #reinforcement_learning(X_train, y_train, X_test, y_test, 30)

    # pentru learning cu reteaua neuronala decomentatie urmatoarele linii
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    # supervized_learning(X_train, y_train, X_test, y_test, 50, None, show_boxplot=True, show_evolution=True)

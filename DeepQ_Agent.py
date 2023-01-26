import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import load_model


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=150, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.q_eval = build_dqn(lr, n_actions, input_dims, 50, 40)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state, verbose=0)

            action = np.argmax(actions)

        return action

    def learn(self, state, action, reward, state_, done):
        states = np.reshape(state, (1, len(state)))
        states_ = np.reshape(state_, (1, len(state_)))
        rewards = np.reshape(reward, (1, 1))
        dones = np.reshape(done, (1, 1))

        q_eval = self.q_eval.predict(states, verbose=0)
        q_next = self.q_eval.predict(states_, verbose=0)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action] = rewards + \
                                         self.gamma * np.max(q_next, axis=1) * dones

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                                                      self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

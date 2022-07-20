# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:46:40 2022

from https://blog.csdn.net/m0_38140207/article/details/106173423
edited by: Phoenix WANG
"""

#from DRL import DRL
from collections import deque
import os
import random
from keras.layers import Dense,Input
from keras import Model
from keras.optimizers import Adam
import numpy as np
import gym
from Drane_Env import Drane_Env
import tensorflow as tf
#ENV_NAME = 'CartPole-v0'
#env = gym.make(ENV_NAME)
#env.seed(123)
#devices = tf.config.expiremental.list_physical_devices('GPU')
#tf.config.expiremental.set_memory_growth(devices[0],True)
env = Drane_Env()

class DQN():
    def __init__(self,env):
        super(DQN, self).__init__()

        self.memory_buffer = deque(maxlen=200)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.env = env
        self.model = self.build_model()

    def load(self):
        if os.path.exists("model/dqn.h5"):
            self.model.load_weights("model/dqn.h5")
        else:
            self.model.load_weights("./dqn.h5")

    def build_model(self):
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation= 'relu')(x)
        x = Dense(8, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='mse', optimizer=Adam(1e-3))

        return model

    def egreedy_action(self, state):
        if np.random.rand() >= self.epsilon:
            return random.randint(0,7)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self,batch):
        data = random.sample(self.memory_buffer, batch)

        # 这里取元组的第1个值，即当前state
        states = np.array([d[0] for d in data])

        # 这里取元组的第4个值，即下一个state
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.model.predict(next_states)
        #print("y value {}, and q value {}".format(y, q))
        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y

    def train(self, episode, batch):
        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation = self.env.reset() # array of float32 (4,1)
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # 一个参数为-1时，reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值
                x = observation.reshape(-1, 4) #shape = (1,4)
                action = self.egreedy_action(x)
                observation, reward, done, _ = self.env.step(action)
                reward_sum += reward
                self.remember(x[0], action, reward, observation, done)
                print(self.env.Step)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    self.update_epsilon()

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))

        self.model.save_weights('dqn1.h5')

        return history
"""
if __name__ == '__main__':
    flag = 0
    model = DQN(env)

    history = model.train(600, 45)
    #model.save_history(history, 'dqn.csv')
    #model.load()
    #model.play()
"""
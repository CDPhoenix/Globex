# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:51:13 2022

@author: Phoenix WANG
"""
from keras.layers import Dense,Input
from keras import Model
from keras.optimizers import Adam
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from DQNmethod import DQN
from Drane_Env import Drane_Env
from Drane_Env_vertical import Drane_Env_vertical

class core():
    def __init__(self,Env,episode,batch,show_animation=False):
        self.model = None
        self.env = Env
        self.episode = episode
        self.batch = batch
        self.status = 0
    
    def LoadingWeight(self):
        self.model.load()
        return 0
    
    def run(self,loading=True):
        HISTORY = []
        for j in range(len(self.env)):
            plt.figure()
            environment = self.env[j]
            self.model = DQN(env=environment)
            if loading == True:
                self.model.load()
            history = self.train()
            ranging = abs(environment.StartingPoint_y - environment.EndingPoint_y)
            observation = environment.reset()
            observation = self.batch_process(observation)
            submodel = self.model.model
            for i in range(int(ranging)):
                action = np.argmax(submodel.predict(observation))
                observation = environment.step(action)[0]
                observation = self.batch_process(observation)
                plt.plot(observation[0][0],observation[0][1],'ro')
            plt.show()
            HISTORY.append(history)
        return HISTORY
    def train(self):
        history = self.model.train(self.episode,self.batch)
        return history
    def batch_process(self,observation):
        observation = [observation.tolist()]
        observation = np.array(observation).astype(np.float32)
        return observation
    def Loss_plot(self,history):
        Loss = history['Loss']
        xlabel = range(len(Loss))
        plt.figure()
        plt.plot(xlabel,Loss)
        plt.show()

Env = [Drane_Env(trainable=True),Drane_Env_vertical(trainable=True)]
#Env = [Drane_Env(trainable=True)]
CORE = core(Env,600,30)
history = CORE.run(loading=False)

for i in range(len(history)):
    CORE.Loss_plot(history[i])

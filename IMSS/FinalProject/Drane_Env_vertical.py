# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:41:17 2022

@author: Phoenix WANG
"""

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class Drane_Env_vertical():
    def __init__(self,trainable=True):
        self.show_animation = False#Show animation
        #Start position
        self.StartingPoint_x = 50.0
        self.StartingPoint_y = 100.0

        #End position
        self.EndingPoint_x = 50.0
        self.EndingPoint_y = 0.0

        #Tolerance
        self.Tolerance = 1.0#Tolerance of landing

        #disturbance
        self.disturbance = [random.uniform(-0.8,0.8),random.uniform(-0.8,0.8)]#random output of disturbance from both env and machine
        #get the motion
        self.step_motion = self.Drane_step_motion()
        self.adjust_motion = self.Adjust_step_motion()
        self.nb_actions = len(self.adjust_motion)
        self.Step = 0
        self.trainable = trainable
        
        
    def Drane_step_motion(self):
        return [0.0,-1.0]
        
    def Adjust_step_motion(self):
        return[[0.4,0.0],
               [-0.4,0.0],
               [0.0,0.4],
               [0.0,-0.4],
               [0.4,0.4],
               [0.4,-0.4],
               [-0.4,0.4],
               [-0.4,-0.4]]

    def step(self,action):
        disturbance = [random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)]#random output of disturbance from both env and machine
        if self.Step == 0:
            self.x = self.StartingPoint_x
            self.y = self.StartingPoint_y
        self.x = self.x + self.step_motion[0]+disturbance[0]+self.adjust_motion[action][0]
        self.y = self.y + self.step_motion[1]+disturbance[1]+self.adjust_motion[action][1]
            
        observation = [self.x,self.y,disturbance[0],disturbance[1]]
        observation = np.array(observation).astype(np.float32)
        
        if self.trainable == True:
            reward = self.reward_calculate(self.x,self.y)
            if self.Step < 50:
                done = False
            else:
                done = True
            _ = None
            self.Step += 1
            return observation,reward,done,_
        else:
            self.Step += 1
            return observation
        
    def reward_calculate(self,x,y):
        distance = abs(x-50)
        reward = 0
        #reward = 10/np.exp(-1*distance)
        distance_to_end = (x-self.EndingPoint_x)**2 + (y-self.EndingPoint_y)**2
        if distance_to_end < self.Tolerance:
            reward += 100
        if distance > 0.3:
            reward = 0
        else:
            reward += 1
        return reward
        
    def reset(self):
        self.x = self.StartingPoint_x
        self.y = self.StartingPoint_y
        self.Step = 0
        observation = [self.StartingPoint_x,self.StartingPoint_y,0.0,0.0]
        observation = np.array(observation).astype(np.float32)
        return observation


# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 10:29:57 2022

@author: 86130
"""
from keras.layers import Dense, Flatten,Conv1D
from keras.models import Sequential
from keras.optimizers import Adam

def EditAgent01(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(144,activation="relu"))
    model.add(Dense(72,activation="relu"))
    model.add(Dense(36,activation="relu"))
    model.add(Dense(actions, activation='linear'))
    return model

def EditAgent02(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(36,activation="relu"))
    model.add(Dense(72,activation="relu"))
    model.add(Dense(36,activation="relu"))
    model.add(Dense(actions, activation='linear'))
    return model

def EditAgent03(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(36,activation="relu"))
    model.add(Dense(72,activation="relu"))
    model.add(Dense(144,activation="relu"))
    model.add(Dense(actions, activation='linear'))
    return model

def EditAgent04(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(256,activation="relu"))
    model.add(Dense(128,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(actions, activation='linear'))
    return model

def EditAgent05(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(144,activation="relu"))
    model.add(Dense(144,activation="relu"))
    model.add(Dense(144,activation="relu"))
    model.add(Dense(actions, activation='linear'))
    return model
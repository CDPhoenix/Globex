"""
Classic cart-pole system. 

The study code is prepared by Professor Xun Huang for the teaching purpose. 
The code is based on OpenAI-Gym. 
2020 July. 

Reference: 
https://gym.openai.com/docs/
"""
# Test 
import gym
import random
import numpy as np
from keras.layers import Dense, Flatten
#from keras.models import Sequential
from keras.models import Sequential
from keras.optimizers import Adam
from Polecart_improvedNetwork import EditAgent01,EditAgent02,EditAgent03,EditAgent04,EditAgent05
# Learning policy
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy
from savingGIF import save_frames_as_gif

env = gym.make('CartPole-v1')

# print states
states = env.observation_space.shape[0]
print('States', states)

# print actions
actions = env.action_space.n
print('Actions', actions)

# Random control
episodes = 10
ob_space,act_space = [],[]# Obeserve the observation space and action space
for episode in range(1,episodes+1):
    # At each begining reset the game 
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:  # When done= True, the game is lost  
        # visualize each step
        env.render()
        # choose a random action
        action = random.choice([0,1])
        # execute the action
        n_state, reward, done, info = env.step(action)
        # keep track of rewards
        score+=reward
        ob_space.append(env.observation_space)
        act_space.append(env.action_space)
    print('episode {} score {}'.format(episode, score))
policy = EpsGreedyQPolicy()
    

# Define a smart agent (a very small network)

def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model0 = agent(env.observation_space.shape[0], env.action_space.n)
model0.summary()



# Agent and compile and training
sarsa = SARSAAgent(model = model0, policy = policy, nb_actions = env.action_space.n)
sarsa.compile('adam', metrics = ['mse'])

#########
sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores = sarsa.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa.save_weights('sarsa_weights.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa.test(env, nb_episodes = 5, visualize= True)
env.close()







"""
####sara1
model1 = EditAgent01(env.observation_space.shape[0], env.action_space.n)
model1.summary()

sarsa1 = SARSAAgent(model = model1, policy = policy, nb_actions = env.action_space.n)
sarsa1.compile('adam', metrics = ['mse'])

sarsa1.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores1 = sarsa1.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa1.save_weights('sarsa_weights1.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa1.test(env, nb_episodes = 5, visualize= True)
env.reset()
#frames1 = []
#for i in range(1000):
#    frames1.append(env.render(mode="rgb_array"))
#    action = env.action_space.sample()
#    _,_,done,_ = env.step(action)
#    if done:
#        break
env.close()
#save_frames_as_gif(frames1, 1)
"""



"""
####sara2
model2 = EditAgent02(env.observation_space.shape[0], env.action_space.n)
model2.summary()

sarsa2 = SARSAAgent(model = model2, policy = policy, nb_actions = env.action_space.n)
sarsa2.compile('adam', metrics = ['mse'])

sarsa2.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores2 = sarsa2.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa2.save_weights('sarsa_weights2.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa2.test(env, nb_episodes = 5, visualize= True)
env.close()
"""

"""
#####sara3
model3 = EditAgent03(env.observation_space.shape[0], env.action_space.n)
model3.summary()

sarsa3 = SARSAAgent(model = model3, policy = policy, nb_actions = env.action_space.n)
sarsa3.compile('adam', metrics = ['mse'])

sarsa3.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores3 = sarsa3.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa3.save_weights('sarsa_weights3.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa3.test(env, nb_episodes = 5, visualize= True)

env.close()
"""


"""
#####sara4
model4 = EditAgent04(env.observation_space.shape[0], env.action_space.n)
model4.summary()

sarsa4 = SARSAAgent(model = model4, policy = policy, nb_actions = env.action_space.n)
sarsa4.compile('adam', metrics = ['mse'])

sarsa4.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores4 = sarsa4.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa4.save_weights('sarsa_weights4.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa4.test(env, nb_episodes = 5, visualize= True)
env.close()
"""


"""
######sara5
model5 = EditAgent05(env.observation_space.shape[0], env.action_space.n)
model5.summary()

sarsa5 = SARSAAgent(model = model5, policy = policy, nb_actions = env.action_space.n)
sarsa5.compile('adam', metrics = ['mse'])

sarsa5.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Then, test the trained model
scores5 = sarsa5.test(env, nb_episodes = 100, visualize=True)
env.close()

# Next, save the model 
sarsa5.save_weights('sarsa_weights5.h5f', overwrite=True)


# load the weights
# sarsa.load_weights('sarsa_weights.h5f')
env = gym.make('CartPole-v1')
_ = sarsa5.test(env, nb_episodes = 5, visualize= True)
env.close()
"""


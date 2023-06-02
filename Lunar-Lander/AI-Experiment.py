import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class agentTrain:

    def __init__(self, env, lr, epsilon): 
        self.env = env
        self.action = env.action_space
        self.num_action = env.action_space.n
        self.observation_space = env.observation_space
        self.num_observation_space = env.observation_space.shape[0]

        self.learning_rate = lr
        self.epsilon = epsilon

        self.model = self.setModel()

        print("Action : ", self.action)
        print("Num action : ", self.num_action)
        print("Observation Space : ", self.observation_space)
        print("Num Observation Space : ", self.num_observation_space)

    def setModel(self):
        l0 = Input(shape = [self.num_observation_space])
        l1 = Dense(512, activation = relu)(l0)
        l2 = Dense(256, activation = relu)(l1)
        l3 = Dense(self.num_action, activation = 'linear')(l2)
        model = tf.keras.Model(inputs = [l0], outputs = [l3])
        model.compile(Adam(learning_rate = self.learning_rate), loss = tf.keras.losses.mean_squared_error)
        return model

    def epsilonGreedyAction(self, state):
        if random.randrange(self.num_action) > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return np.random.randint(self.num_action)

    def agentTrain(self, episodes):
        for episode in range(episodes):
            print("Episode : ", episode)
            state = self.env.reset()
            print("State Shape: ", state[0].shape)
            rewardEpisode = 0
            steps = 500
            state = np.reshape(state[0], [1, self.num_observation_space])

            for step in range(steps):
                action = self.epsilonGreedyAction(state)
                print("Action = ", action)
                nextState = env.step(action)[0]
                reward = env.step(action)[1]
                status = env.step(action)[2]
                info = env.step(action)[3]
                self.env.render()

                nextState = np.reshape(nextState, [1, self.num_observation_space])

                rewardEpisode += reward
                state = nextState

                if status:
                    break

            
if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="human") 
    
    lr = 0.001
    epsilon = 1
    trainAgent = agentTrain(env, lr, epsilon)
    trainAgent.agentTrain(10)

    env.close()
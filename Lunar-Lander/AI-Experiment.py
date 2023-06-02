import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class agentTrain:

    def __init__(self, env): 
        self.action = env.action_space # agent policy that uses the observation and info
        self.num_action = env.action_space.n
        self.observation_space = env.observation_space
        self.num_observation_space = env.observation_space.shape[0]

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
        model.compile(Adam(learning_rate = 0.001), loss = tf.keras.losses.mean_squared_error)
        return model


if __name__ == '__main__':
    env = gym.make("LunarLander-v2") #, render_mode="human"

    trainAgent = agentTrain(env)

    env.close()
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class agentTrain:

    def __init__(self, env, lr, epsilon, epsilonDecay, gamma): 
        self.env = env
        self.action = env.action_space
        self.num_action = env.action_space.n
        self.observation_space = env.observation_space
        self.num_observation_space = env.observation_space.shape[0]

        self.learning_rate = lr
        self.epsilon = epsilon
        self.minEpsilon = 0.01
        self.decay = epsilonDecay
        self.discount = gamma

        self.count = 0
        self.rewardList = []
        self.buffer = deque(maxlen = 1000000000)
        self.batchSize = 64

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
    
    def addToBuffer(self, state, action, reward, nextState, status):
        self.buffer.append((state, action, reward, nextState, status))

    def counterUpdate(self):
        self.count += 1
        step = 5
        self.count = self.count % step

    def agentTrain(self, episodes):
        for episode in range(episodes):
            print("Episode : ", episode)
            print("Epsilon : ", self.epsilon)
            state = self.env.reset()
            # print("State Shape: ", state[0].shape)
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
                self.addToBuffer(state, action, reward, nextState, status)

                rewardEpisode += reward
                state = nextState

                self.counterUpdate()
                self.updateModel()

                if status:
                    break
            print("Reward for Episode : ", rewardEpisode)
            self.rewardList.append(rewardEpisode)

            if self.epsilon > self.minEpsilon:
                self.epsilon *= self.decay
            lastRewardMean = np.mean(self.rewardList[-100:])

            if lastRewardMean > 200:
                print("Training Complete")
                break

            self.model.save('model.h5', overwrite = True)

    def attrFromSample(self, sample):
        states = np.squeeze(np.squeeze(np.array([i[0] for i in sample])))
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        nextState = np.squeeze(np.array([i[3] for i in sample]))
        status = np.array([i[4] for i in sample])
        return states, actions, rewards, nextState, status

    def updateModel(self):
        if len(self.buffer) < self.batchSize or self.count != 0:
            return

        randomSample = random.sample(self.buffer, self.batchSize)
        # print(randomSample[0][1])
        state, action, reward, nextState, status = self.attrFromSample(randomSample)

        target = reward + self.discount * (np.max(self.model.predict_on_batch(nextState), axis=1)) * (1 - status)
        targetVector = self.model.predict_on_batch(state)
        indexes = np.array([i for i in range(self.batchSize)])
        targetVector[[indexes], [action]] = target

        self.model.fit(state, targetVector, epochs=1, verbose=0)

            
if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="human") 
    
    lr = 0.001
    epsilon = 1
    epsilonDecay = 0.99
    gamma = 0.99

    trainAgent = agentTrain(env, lr, epsilon, epsilonDecay, gamma)
    trainAgent.agentTrain(2000)
    trainAgent.updateModel()

    env.close()
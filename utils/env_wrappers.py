import gym
import numpy as np
import cv2

class AtariWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = []

    def reset(self):
        obs = self.env.reset()
        obs = self.preprocess(obs)
        self.frames = [obs] * self.k
        return np.stack(self.frames, axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)
        self.frames.pop(0)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, done, info

    def preprocess(self, obs):
        # Convert to grayscale and resize to 84x84
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs / 255.0

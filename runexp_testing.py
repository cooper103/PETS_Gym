#!/usr/bin/env python3

#algorithm for PETS probablistic dynamics model with ensembles
import sys
import numpy as np
from collections import deque, namedtuple
import random as rand
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from CEM_with_PETS import CEM
from PE_testing import PE
from TSinf_testing import TSinf
import time 
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'next_state'))
episode_rewards = [] 
def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)

    if show_result:
            plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def sample_with_replacement(self, batch_size):
        indices = np.random.choice(self.length(), size=batch_size, replace=True)
        result = [self.memory[i] for i in indices]
        return result
    
    def length(self):
        return len(self.memory)



if __name__ == '__main__':
    K = 1000 #number of trials
    B = 5 #number of bootstrapped networks
    T = 30 #task horizon of CEM planner also how long particles will be proagated
    #P = 20 #number of particles 
    env = gym.make("CartPole-v1")

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)



    dim_theta = (env.observation_space.shape[0]+1)*env.action_space.n
    CEMPlanner = CEM(T, 2, env)

    input_dims = 1 + env.observation_space.shape[0]
    output_dims = 2 * env.observation_space.shape[0]
    #print('output dims', output_dims)
    #print('input dims', input_dims)
    #
    #print("input dims", input_dims)
    ProbablisticEnsemble = PE(input_dims, output_dims)
    TSinfPropagate = TSinf(T)
    dataset = Memory(10000)

    while dataset.length() < 512:
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, _, done, _ , _= env.step(action)
            transition = Transition(state, action, next_state)
            #print(transition)
            dataset.push(transition)
            state = next_state
    print("this version uses large CEM samples and trains ecah timestep")
    for i in range(K):
       # time1 = time.time()
        #time2 = time.time()
       # elapsed = str(time2-time1)
        #print("training time elapsed, " + elapsed)
        state, _ = env.reset()
        done = False
        rewards = 0
        while not done:
            ProbablisticEnsemble.train(dataset)
            #time1= time.time()
            action = CEMPlanner.optimal_action(state, ProbablisticEnsemble, TSinfPropagate, env) 
            next_state, reward, done, _ , _ = env.step(action)
            transition = Transition(state, action, next_state)
            state = next_state
            dataset.push(transition)
            rewards += reward
            #time2 = time.time()
            #elapsed = str(time2-time1)
            #print("running exp time elapsed, " + elapsed)
        episode_rewards.append(rewards)
        
        plot_durations()
        print("trial: ", i, " reward: ", rewards)
        
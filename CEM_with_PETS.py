#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from TSinf_testing import TSinf
import time


"""
"""

def get_reward(state, action):
    if abs(state[0]) > 2.4:
        reward = 0
    elif abs(abs(state[2]) > .2095):
        reward = 0
    else:
        reward = 1
    return reward


def evaluate_actions(states ,actions):
    #to do: determine how TSinf will know if it has predicted a temrinal
    #state and also account for this in thte reward evaluation of action 
    #sequences
    reward = 0
    #print("states, ", states)
    done = False
    assert len(states) == len(actions)
    for t in range(len(states)):
        r = get_reward(states[t], actions[t])
        if r == 0:
            done = True
        if done:
            reward += 0
        else:
            reward += r
    reward /= len(actions)#average reward of each sequence
    return reward


class DeterministicDiscreteActionLinearPolicy(object):
    def __init__(self, theta, num_obs, num_act):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = num_obs
        n_actions = num_act
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)
        #print("W", self.W, "Shape: ", self.W.shape)
        
    def act(self, observation):
        """
        returns the best action
        """
        #print("observation is: ", observation[0].shape)
        #print("W is: ", self.W.shape)
        y = np.dot(observation, self.W)+ self.b
        #print("Y is ", y)
        #print(y.shape)
        a = np.argmax(y)
        #print("action is ", a)
        return a


class CEM():
    def __init__(self, task_horizon, num_actions, env):
        self.env = env
        self.dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
        self.task_horizon = task_horizon
        self.num_actions = num_actions
        self.num_iters = 10
        self.theta_mean = np.zeros(self.dim_theta)
        self.theta_std = np.ones(self.dim_theta)
        self.batch_size = 100
        self.n_elite = 10
        self.alpha = 0.9

    def make_policy(self, theta):
        return DeterministicDiscreteActionLinearPolicy(theta, self.env.observation_space.shape[0], self.env.action_space.n)
    
    def generate_actions(self):
        thetas = np.random.multivariate_normal(mean=self.theta_mean, cov=np.diag((np.array(self.theta_std**2, size=self.task_horizon))))
    
    def evaluate_thetas(self, thetas, state, ProbablisticEnsemble, TSinfPropagate, env):
        reward = 0
        policy = DeterministicDiscreteActionLinearPolicy(thetas, self.env.observation_space.shape[0], self.env.action_space.n)
        action_sequence, state_sequence = TSinfPropagate.propagate_particle(state, policy, ProbablisticEnsemble, env)
        reward = evaluate_actions(state_sequence, action_sequence)
        return reward

    def optimal_action(self, state, ProbablisticEnsemble, TSInfPropagate, env):
        self.theta_mean = np.zeros(self.dim_theta)
        self.theta_std = np.ones(self.dim_theta)
        for i in range(self.num_iters):
            thetas = np.random.multivariate_normal(mean=self.theta_mean, cov=np.diag(np.array(self.theta_std**2)), size=self.batch_size)
            rewards = np.zeros(self.batch_size)
            i = 0
            for theta in thetas:
                rewards[i] = self.evaluate_thetas(theta, state, ProbablisticEnsemble, TSInfPropagate, env)
                i += 1

            elite_inds = np.argsort(rewards)[-self.n_elite:]
            elite_thetas = thetas[elite_inds]
            self.theta_mean = elite_thetas.mean(axis=0) * self.alpha + (1-self.alpha)*self.theta_mean
            self.theta_std = elite_thetas.std(axis=0) * self.alpha + (1-self.alpha)*self.theta_std

        self.theta_mean = elite_thetas.mean(axis=0)
        self.theta_std = elite_thetas.std(axis=0)
        action = DeterministicDiscreteActionLinearPolicy(self.theta_mean, self.env.observation_space.shape[0], self.env.action_space.n).act(state)
        return action
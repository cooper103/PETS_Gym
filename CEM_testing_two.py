#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
import time

"""
example code for a cross-entropy method implentaiton that solves the cartpole task. This version is my own code.
"""

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
        T = 20
        P = 20
        self.dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
        self.task_horizon = task_horizon
        self.num_actions = num_actions
        self.num_iters = 5
        self.num_particles = 20 #propagate 20 particles for each action seq. 
        self.theta_mean = np.zeros(self.dim_theta)
        self.theta_std = np.ones(self.dim_theta)
        self.batch_size = 500
        self.n_elite = 3
        self.alpha = 0.99

    def make_policy(self, theta):
        return DeterministicDiscreteActionLinearPolicy(theta, self.env.observation_space.shape[0], self.env.action_space.n)
    

    def generate_actions(self):
        thetas = np.random.multivariate_normal(mean=self.theta_mean, cov=np.diag((np.array(self.theta_std**2, size=self.task_horizon))))

    def do_episode(self, policy, env, num_steps=500, discount=1.0, render=False):
        disc_total_rew = 0
        ob = self.env.reset()[0]
        #print("observation", ob)
        for t in range(num_steps):
            a = policy.act(ob)
            ob, reward, done, _info, _ = env.step(a)
            #print("this observation is ", ob)
            disc_total_rew += reward * discount**t
            if render:
                env.render()
                #time.sleep(0.1)
            if done: break
        env.close()
        return disc_total_rew, t
    
    def evaluate_thetas(self, thetas):
        reward = 0
        policy = DeterministicDiscreteActionLinearPolicy(thetas, self.env.observation_space.shape[0], self.env.action_space.n)
        reward, _ = self.do_episode(policy, env)
        return reward

    def update_distribution(self):
        pass

    def optimal_action(self, state):
        self.theta_mean = np.zeros(self.dim_theta)
        self.theta_std = np.ones(self.dim_theta)
        for i in range(self.num_iters):
            thetas = np.random.multivariate_normal(mean=self.theta_mean, cov=np.diag(np.array(self.theta_std**2)), size=self.batch_size)
            #print(thetas)
            #print(thetas.shape)
            #rewards = np.array(map(self.evaluate_thetas, thetas))
            rewards = np.zeros(self.batch_size)
            i = 0
            for theta in thetas:
                rewards[i] = self.evaluate_thetas(theta)
                i += 1

            self.update_distribution()
            elite_inds = np.argsort(rewards)[-self.n_elite:]
            elite_thetas = thetas[elite_inds]
            self.theta_mean = elite_thetas.mean(axis=0) * self.alpha + (1-self.alpha)*self.theta_mean
            self.theta_std = elite_thetas.std(axis=0) * self.alpha + (1-self.alpha)*self.theta_std

    # Update theta_mean, theta_std
        self.theta_mean = elite_thetas.mean(axis=0)
        self.theta_std = elite_thetas.std(axis=0)
        
        return self.theta_mean

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    cem = CEM(30, 2, env)
    ob = env.reset()
    theta_mean = cem.optimal_action(ob)
    print("final theta mean: ", theta_mean)
    rewards = 0
    for i in range(30):
        _, reward = cem.do_episode(cem.make_policy(theta_mean), env, render =True)
        rewards += reward
    print(rewards/30)
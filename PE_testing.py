#!/usr/bin/env python3

#contains the class for ensemble of probablistic neural networks. There will be
#5 bootstrapped networks used to propagate the 20 state particles used in evaluating
#a CEM action sequence. Each bootstrapped network will trained on unique data
#sampled with replaced from the overall data distribution D. The overall 
#predictive probablility distribution f~ is parameterized by the average
#of parameters for the 5 bootstrapped networks. The networks will be trained
#after each trial 

#for this use case, the probablistic network input will be the state and action
#the neural network output will be the means and variances of the predicted 
#next state. The variance outputs will be treated as log variances and will be
#passed through an exponenetial funciton before going into the loss funciton
#to ensure that the output is always positive. Before going into the exp
#function these values are also upper and lower bounded such that the output 
#variance cannot be higher or lower than the lowest/highest values in the training
#data. 

#the loss function will be Gaussian Negative Log Likelihood Loss, which is
#explained clearly in the paper and in pytorch documentation

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state'))

#probablistic NN that outputs mean and diagonal of covariance matrix of Gaussian
class P(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(P, self).__init__()
        #print(input_dims)
        self.fc1 = nn.Linear(input_dims, 500)
        self.swish1 = nn.SiLU()
        self.fc2 = nn.Linear(500, 500)
        self.swish2 = nn.SiLU()
        self.fc3 = nn.Linear(500, output_dims)
        #self.opt = Adam(cfglr=1e-3)
        #self.loss_function = nn.GaussianNLLLoss()

    def forward(self, x, in_batches=True,is_training=True):
        out = self.fc1(x)
        out = self.swish1(out)
        out = self.fc2(out)
        out = self.swish2(out)
        out = self.fc3(out)
        sp = nn.Softplus()
        if in_batches:
            logvar = out[:, 4:8]
        else:
            logvar = out[4:8]
        var = torch.exp(logvar)
        if in_batches:
            out[:,4:8] = var
        else:
            out[4:8] = var
        return out
    """
        code to have a max and min logvariance. this version causes some minor erros
        so will worry about implementing later if it becomes an issue
        if is_training:
            try:
                self.max_logvar = int(torch.maximum(self.max_logvar, torch.max(logvar)))
                self.min_logvar = int(torch.minimum(self.min_logvar, torch.min(logvar)))
            except(UnboundLocalError):
                self.min_logvar = torch.min(logvar, 0)[0]
                self.max_logvar = torch.max(logvar, 0)[0]
                
        if is_training:
            logvar = self.max_logvar - sp(self.max_logvar - logvar)
            logvar = self.min_logvar - sp(logvar - self.min_logvar)
    """
        

    def train(self, dataset,opt,loss_fn):
        #generate unique dataset by sampling with replacement 
        #unique_dataset = dataset.sample_with_replacement(dataset.length())
        unique_dataset = dataset.sample_with_replacement(256)
        batch = Transition(*zip(*unique_dataset))
        #target =  next_state
        #next state values from our dataset
        target = torch.tensor(batch.next_state, dtype = torch.float32)
        state = np.array(batch.state)
        action = np.array([batch.action]).T
        #print(state.shape)
        #print(action.shape)
        state_action = np.concatenate((state, action), axis=1)
        #print("state aciton", state_action)
        state_action = torch.tensor(state_action,dtype=torch.float32)
        #input = mean prediction from nn
        #output mean prediction depending on current state and action
        
        nn_output = self.forward(state_action)
        input_vals = nn_output.index_select(1, torch.range(start=0,end=3, dtype=torch.int32))
        var = nn_output.index_select(1, torch.range(start=4,end=7, dtype=torch.int32))
        #print(nn_output)
        #print(var)
        #var = variance prediction from nn
        #output variance prediciton depending on current state and action

        loss = loss_fn(input_vals, target, var)
        opt.zero_grad()
        loss.backward()
        opt.step()

class PE():
    def __init__(self, input_dims, output_dims):
        #print("PE constructor, inputs_dims are ", input_dims)
        self.P1 = P(input_dims, output_dims)
        self.P2 = P(input_dims, output_dims)
        self.P3 = P(input_dims, output_dims)
        self.P4 = P(input_dims, output_dims)
        self.P5 = P(input_dims, output_dims)
        self.PE_array = [self.P1, self.P2, self.P3, self.P4, self.P5]

        lr=1e-4

        self.loss_fn = nn.GaussianNLLLoss()
        self.opt1 = Adam(self.P1.parameters(), lr=lr)
        self.opt2 = Adam(self.P2.parameters(), lr=lr)
        self.opt3 = Adam(self.P3.parameters(), lr=lr)
        self.opt4 = Adam(self.P4.parameters(), lr=lr)
        self.opt5 = Adam(self.P5.parameters(), lr=lr)
        self.opt_arr = [self.opt1, self.opt2, self.opt3, self.opt4, self.opt5]

    def train(self, dataset):
        for i  in range(len(self.PE_array)):
            self.PE_array[i].train(dataset, self.opt_arr[i], self.loss_fn)





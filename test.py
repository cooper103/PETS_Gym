#!/usr/bin/env python3

#algorithm for PETS probablistic dynamics model with ensembles
import sys
import numpy as np
from collections import deque, namedtuple
import random as rand
import torch
import matplotlib.pyplot as plt

from CEM_testing import CEM
from PE_testing import PE
from TSinf_testing import TSinf
from tb3_env_simple_nav_testing import turtlebot3Env
from runexp_testing import Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state'))

state = np.full((26), 4)
action = 2

Probablistic_Ensemble = PE(27, 52)
dataset = Memory(1000)
for i in range(256):
    next_state = np.random.normal(loc=5, scale = 1, size=26)
    trans = Transition(state, action, next_state)
    dataset.push(trans)
for i in range(300):
    Probablistic_Ensemble.train(dataset)

state_action = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2])
state_action = torch.tensor(state_action, dtype=torch.float32)
print(Probablistic_Ensemble.PE_array[0].forward(state_action, in_batches=False))

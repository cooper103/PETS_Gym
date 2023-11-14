#!/usr/bin/env python3

#will generate 20 state particles, each of which has a bootstrap index that 
#remains constant with time, which separates aleatoric and epistemic uncertainty
import torch
import numpy as np
class TSinf:
    def __init__(self, T):
        self.T = T
        self.i = 0
    def propagate_particle(self, s0, policy, ProbablisticEnsemble, env):
        state = s0
        action_sequence = np.zeros(self.T)
        #print('shape of obs space', env.observation_space.shape[0])
        x = np.arange((self.T * env.observation_space.shape[0]))
        x = x.reshape((self.T, env.observation_space.shape[0]))
        state_sequence = np.zeros_like(x)
        #print("State Sequence, ", state_sequence)
        Pb = ProbablisticEnsemble.PE_array[self.i%len(ProbablisticEnsemble.PE_array)]
        Pb = ProbablisticEnsemble.PE_array[0]
        for step in range(self.T):
            state_sequence[step] = state
            action = policy.act(state)
            action_sequence[step] = action
            #print('state', state)
            #print('action', action)
            state_action = np.concatenate((state, np.array([action])))
            state_action = torch.tensor(state_action,dtype=torch.float32)
            nn_output = Pb.forward(state_action, in_batches=False,is_training=False)
            nn_output = nn_output.cpu().detach().numpy()
            n_state = nn_output[0:4]
            #n_state = np.random.normal(loc=nn_output[0:4], scale=nn_output[4:8])
            state = n_state
        self.i += 1
        return action_sequence, state_sequence

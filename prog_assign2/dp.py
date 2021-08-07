from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    V = np.array(initV)
    num_states = env.spec.nS
    num_actions = env.spec.nA
    Q = np.zeros((num_states,num_actions))
    reward = env.R
    TD = env.TD

    delta = float('inf')
    while delta >= theta:
        delta = 0.0
        for state in range(num_states):
            old_v = V[state]
            new_v = 0
            for action in range(num_actions):
                sum_values = 0
                for sPrime in range(num_states):
                    sum_values += TD[state,action,sPrime] * (reward[state,action,sPrime] + env.spec.gamma * V[sPrime])
                new_v += pi.action_prob(state,action) * sum_values
            delta = max(delta, abs(new_v-old_v))
            V[state] = new_v

    for state in range(num_states):
        for action in range(num_actions):
            for sPrime in range(num_states):
                Q[state][action] += TD[state,action,sPrime] * (reward[state,action,sPrime] + env.spec.gamma * V[sPrime])  

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    V = np.array(initV)
    num_states = env.spec.nS
    num_actions = env.spec.nA
    reward = env.R
    TD = env.TD
    
    delta = float('inf')
    while delta >= theta:
        delta = 0.0
        for state in range(num_states):
            old_v = V[state]
            new_v = np.NINF
            for action in range(num_actions):
                sum_values = 0
                for sPrime in range(num_states):
                    sum_values += TD[state,action,sPrime] * (reward[state,action,sPrime] + env.spec.gamma * V[sPrime])
                new_v = max(new_v, sum_values)
            delta = max(delta, abs(new_v-old_v))
            V[state] = new_v
    
    # create deterministic policy
    opt_actions = np.zeros(num_states,dtype=int)
    for state in range(num_states):
        action_values  = np.zeros(num_actions)
        for action in range(num_actions):
            sum_values = 0
            for sPrime in range(num_states):
                sum_values += TD[state,action,sPrime] * (reward[state,action,sPrime] + env.spec.gamma * V[sPrime])
            action_values[action] = sum_values
        opt_actions[state] = np.argmax(action_values)
    
    pi = optimal_policy(opt_actions)

    return V, pi

class optimal_policy(Policy):
    def __init__(self, opt_actions):
        self.opt_actions = opt_actions

    def action_prob(self, state:int, action:int):
        if self.opt_actions[state] == action: 
            return 1
        else:
            return 0
    
    def action(self, state:int):
        return self.opt_actions[state]

    

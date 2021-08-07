from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    V = np.array(initV)
    for episode in trajs:
        T = float('inf')
        t = 0
        tau = 0
        state = []
        reward = []
        state.append(episode[t][0])
        while tau != T - 1: 
            if t < T:
                reward.append(episode[t][2])
                state.append(episode[t][3])
                if t == len(episode) - 1:
                    T = t + 1
            tau = t - n + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (env_spec._gamma ** (i - tau - 1)) * reward[i - 1]
                if tau + n < T:
                    G += ((env_spec._gamma ** n) * V[state[tau + n]])
                V[state[tau]] += (alpha * (G - V[state[tau]]))
            t += 1
        
    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    Q = np.zeros((env_spec.nS, env_spec.nA))
    Q = initQ
    policy = optimal_Policy(Q)
    pi = np.zeros(env_spec.nS)

    for episode in trajs:
        T = float('inf')
        tau = 0
        t = 0
        state = []
        action = []
        reward = []
        state.append(episode[t][0])
        action.append(episode[t][1])
        while tau != T - 1:
            if t < T:
                reward.append(episode[t][2])
                state.append(episode[t][3])
                if t == len(episode) - 1:
                    T = t + 1
                else:
                    action.append(episode[t + 1][1])
            tau = t - n + 1
            if tau >= 0:
                rho = 1
                G = 0
                for i in range(tau + 1, min(tau + n, T - 1) + 1):
                    rho = rho * (policy.action_prob(state[i], action[i]) / bpi.action_prob(state[i], action[i]))
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (env_spec._gamma ** (i - tau - 1)) * reward[i - 1]

                if tau + n < T:
                    G += ((env_spec._gamma ** n) * Q[state[tau + n]][action[tau + n]])

                Q[state[tau]][action[tau]] += alpha * rho * (G - Q[state[tau]][action[tau]])

            t += 1

    pi = policy
    return Q, pi

class optimal_Policy(object):
    def __init__(self, Q):
        self.Q = Q

    def action_prob(self, state: int, action: int) -> float:
        if np.argmax(self.Q[state]) == action:
            return 1
        else:
            return 0

    def action(self, state: int) -> int:
        return np.argmax(self.Q[state])

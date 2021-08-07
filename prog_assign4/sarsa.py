import numpy as np
import math
import gym
import functools

def create_one_tiling(feat_range, bins, offset, step_size):
    #create 1 tiling for 1 feature
    return np.linspace(feat_range[0], feat_range[1] + step_size, bins+1)[1:-1] - offset


def create_mutiple_tilings(feat_ranges, number_tilings, bins, offsets, step_sizes):
    """
    feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings; example: 3tilings
    bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
    offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
    """
    tilings = []
    # for each tiling
    for tile in range(number_tilings):
        bin = bins[tile]
        offset = offsets[tile]
        tiling = []
        # for each feature dimension
        for feat_idx in range(len(feat_ranges)):
            feat_range = feat_ranges[feat_idx]
            # tiling for 1 feature
            feat_tiling = create_one_tiling(feat_range, bin[feat_idx], offset[feat_idx], step_sizes[feat_idx])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)

def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """
    num_dims = len(feature)
    feat_codings = []
    for tile in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat = feature[i]
            tiling = tile[i]  # tiling on that dimension
            coding = np.digitize(feat, tiling)
            feat_coding.append(coding)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        super().__init__()
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        dimensions = len(state_low)
        state_ranges = []
        bins = []
        offsets = []
        for i in range(dimensions):
            state_range = [state_low[i], state_high[i]]
            state_ranges.append(state_range)

        for i in range(num_tilings):
            bin = []
            offset = []
            for j in range(dimensions):
                # Number of tiles along jth dimension for a given tiling
                bin.append(math.ceil((state_high[j] - state_low[j]) / tile_width[j]))
                #print(bin)
                # Offset along jth dimension for a given tiling 
                offset.append(i * tile_width[j] / num_tilings)
                #print(offset)

            bins.append(bin)
            offsets.append(offset)
            
        get_tilings = create_mutiple_tilings(state_ranges, num_tilings, bins, offsets, self.tile_width)
        self.tilings = get_tilings
        #initialize weights
        #self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in self.tilings]
        self.num_tiles = functools.reduce(lambda a,b : a*b, bins[0])
        #print(self.num_tiles)
        self.width = bins[0][0]
        #print(self.width)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        #raise NotImplementedError()
        dim = self.num_actions * self.num_tilings * self.num_tiles
        return dim 
        

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        #raise NotImplementedError()
        state_values = np.zeros(self.feature_vector_len())
        #if done is true, then return 0^d
        if done: 
            return state_values
        else: 
            get_codings = get_tile_coding(s, self.tilings)
            for i in range(self.num_tilings): 
                coding = get_codings[i]
                index = (a * self.num_tilings * self.num_tiles) + (i * self.num_tiles) + (self.width * (coding[0]) + coding[1])
                #print(index)
                state_values[index] = 1
        return state_values
    
def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len())) #num_actions * num_tilings * num_tiles

    #TODO: implement this function
    #raise NotImplementedError()
    for episode in range(num_episode): 
        S0 = env.reset()
        done = False
        action = epsilon_greedy_policy(S0, done, w)
        x = X(S0, done, action)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0
        while not done: 
            observation, reward, done, info = env.step(action)
            new_action = epsilon_greedy_policy(observation, done, w)
            x_dash = X(observation, done, new_action)
            Q = np.dot(w, x)
            Q_dash = np.dot(w, x_dash)
            delta = reward + gamma * Q_dash - Q
            z = gamma * lam * z + (1-gamma*lam*alpha*np.dot(z, x)) * x
            w = w + alpha*(delta + Q - Q_old)*z - alpha * (Q-Q_old) * x
            Q_old = Q_dash
            x = x_dash
            action = new_action
    
    return w 


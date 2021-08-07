import numpy as np
from algo import ValueFunctionWithApproximation
import math
import gym
#reference the code from https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b

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

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method **Create the tiling and coding here**
        super().__init__()
        self.state_low = state_low
        self.state_high = state_high
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
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in self.tilings]
        self.weights = [np.ones(shape=state_size) for state_size in self.state_sizes]

    def __call__(self, s):
        # TODO: implement this method
        value = 0
        codings = get_tile_coding(s, self.tilings)
        for i in range(self.num_tilings):
            value = value + self.weights[i][tuple(codings[i])]
        return value

    def update(self, alpha, G, s_tau):
        # TODO: implement this method
        codings = get_tile_coding(s_tau, self.tilings)
        temp = self.__call__(s_tau)
        delta = alpha * (G - temp)
        for i in range(self.num_tilings):
            self.weights[i][tuple(codings[i])] += delta
        return None
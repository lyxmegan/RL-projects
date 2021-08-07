import numpy as np

def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """
    
    return np.linspace(feat_range[0], feat_range[1], bins+1)[1:-1] + offset


def create_multiple_tilings(feat_ranges, num_tilings, bins, offsets):
        """
        create multiple tilings for multiple features
        feat_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
        num_tilings: number of tilings; example: 3 tilings
        bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
        offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
        """
        tilings = []
        # for each tiling
        for tile_idx in range(number_tilings):
            tiling_bin = bins[tile_idx]
            tiling_offset = offsets[tile_idx]

            tiling = []
            # for each feature dimension
            for feat_idx in range(len(feature_ranges)):
                feat_range = feature_ranges[feat_idx]
                # one tiling for 1 feature
                feat_tiling = create_tiling(feat_range, tiling_bin[feat_idx], tiling_offset[feat_idx])
                tiling.append(feat_tiling)
            tilings.append(tiling)
        print(tilings)
        
        return np.array(tilings)

feature_ranges = [[-1, 1], [2, 5]]  # 2 features
number_tilings = 3
bins = [[10, 10], [10, 10], [10, 10]]  # each tiling has a 10*10 grid
offsets = [[0, 0], [0.2, 1], [0.4, 1.5]]

tilings = create_multiple_tilings(feature_ranges, number_tilings, bins, offsets)

print(tilings.shape)  # # of tilings X features X bins

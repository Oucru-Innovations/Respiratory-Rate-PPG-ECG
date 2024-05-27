import numpy as np

def simple_average(*estimations):
    return np.mean(estimations, axis=0)

def weighted_average(estimations, weights):
    return np.average(estimations, axis=0, weights=weights)

def median_fusion(*estimations):
    return np.median(estimations, axis=0)

def mode_fusion(*estimations):
    from scipy.stats import mode
    return mode(estimations, axis=0)[0]

def voting_fusion(*estimations):
    from scipy.stats import mode
    return mode(estimations, axis=0)[0]

def combine_estimations(estimations, method='average', weights=None):
    if method == 'average':
        return simple_average(*estimations)
    elif method == 'weighted':
        if weights is None:
            weights = np.ones(len(estimations))
        return weighted_average(estimations, weights)
    elif method == 'median':
        return median_fusion(*estimations)
    elif method == 'mode':
        return mode_fusion(*estimations)
    elif method == 'voting':
        return voting_fusion(*estimations)
    else:
        raise ValueError(f"Unknown combination method: {method}")


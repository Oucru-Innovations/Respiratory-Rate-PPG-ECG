import numpy as np
from scipy.stats import mode

def simple_average(*estimations):
    """Compute the simple average of multiple estimations.

    Parameters
    ----------
    *estimations : array-like
        A variable number of arrays to be averaged.

    Returns
    -------
    float or ndarray
        The average of the provided estimations.
    """
    return np.mean(estimations, axis=0)

def weighted_average(estimations, weights=None):
    """Compute the weighted average of estimations.

    Parameters
    ----------
    estimations : array-like
        Array of estimations to be combined.
    weights : array-like, optional
        Weights for each estimation, by default None (equal weights).

    Returns
    -------
    float or ndarray
        Weighted average of the estimations.
    """
    if weights is None:
        weights = np.ones(len(estimations))
    return np.average(estimations, axis=0, weights=weights)

def median_fusion(*estimations):
    """Combine multiple estimations using median fusion.

    Parameters
    ----------
    *estimations : array-like
        A variable number of arrays to combine.

    Returns
    -------
    float or ndarray
        The median of the provided estimations.
    """
    return np.median(estimations, axis=0)

def mode_fusion(*estimations):
    """Combine multiple estimations using mode fusion.

    Parameters
    ----------
    *estimations : array-like
        A variable number of arrays to combine.

    Returns
    -------
    float or ndarray
        The mode of the provided estimations.
    """
    return mode(estimations, axis=0)[0]

def voting_fusion(*estimations):
    """Combine multiple estimations using voting (mode-based) fusion.

    Parameters
    ----------
    *estimations : array-like
        A variable number of arrays to combine.

    Returns
    -------
    float or ndarray
        The mode of the provided estimations, representing the most common values.
    """
    return mode(estimations, axis=0)[0]

def combine_estimations(estimations, method='average', weights=None):
    """Combine multiple estimations using a specified fusion method.

    Parameters
    ----------
    estimations : list of array-like
        List of estimations to combine.
    method : str, default='average'
        Combination method to use, one of {'average', 'weighted', 'median', 'mode', 'voting'}.
    weights : array-like, optional
        Weights for the 'weighted' method, by default None (equal weights).

    Returns
    -------
    float or ndarray
        Combined estimation result based on the chosen method.

    Raises
    ------
    ValueError
        If an unknown combination method is specified.
    """
    if method == 'average':
        return simple_average(*estimations)
    elif method == 'weighted':
        return weighted_average(estimations, weights=weights)
    elif method == 'median':
        return median_fusion(*estimations)
    elif method == 'mode' or method == 'voting':
        return mode_fusion(*estimations)
    else:
        raise ValueError(f"Unknown combination method: {method}")

import numpy as np


def featureScaling(X):
    """Scales features."""
    stdev_Of_Each_Feature = np.std(X, axis=0)
    mean_Of_Each_Feature = X.mean(0)
    scaled_X = (X - mean_Of_Each_Feature) / stdev_Of_Each_Feature

    """These are for checking.
    # range_Of_Features = np.ptp(X, axis=0)
    # mean_ = X.mean(0)
    # max_ = np.max(X, axis=0)
    # min_ = np.min(X, axis=0)
    # stdev = np.std(X, axis=0)
    """

    # scaled_X_ver2 = (X - mean_Of_Each_Feature) / range_Of_Features # This one is to test different scaling for features.
    # return scaled_X_ver2
    return scaled_X
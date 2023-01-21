import numpy as np

def init_variables():
    """
        init model variables (weight - bias)
    """
    weights = np.random.normal(size=2)
    bias = 0 
    

    return weights, bias


def get_dataset():
    """
        method to get a dataset
    """
    # number of row per class
    row_per_class = 5

    #generate rows 
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])
    
    features = np.vstack([sick, healthy])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    return features, targets 


if __name__ == "__main__":
    features, targets = get_dataset()
    init_variables()
    pass


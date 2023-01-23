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

def pre_activation(features, wheights, bias):
    """
        pre-activation
    """
    return(np.dot(features, wheights) + bias)


def activation(z):
    """
        activation
    """
    return 1 / (1 + np.exp(-z))



if __name__ == "__main__":
    #rows
    features, targets = get_dataset()
    #variable
    weights, bias = init_variables()
    # calculate the pre-activation
    z = pre_activation(features, weights, bias)
    a = activation(z)
    print(a)
    pass


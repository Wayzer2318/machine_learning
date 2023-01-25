import numpy as np
import matplotlib.pyplot as plt

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
    row_per_class = 100

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

def d_activation(z):
    return activation(z) * (1 - activation(z))

def predict(features, weights, bias):
    """
    """
    z = pre_activation(features, weights, bias)
    y = activation(z)
    return np.round(y)

def cost(predictions, targets):
    """
    """
    return np.mean((predictions - targets)**2)

def train(features, targets, weights, bias):
    """
    """
    epoch = 100
    learning_r = 0.1
    #print current accurency
    predictions = predict(features, weights, bias)
    print("Accurency: ", np.mean(predictions == targets))
    # Plot points
    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()
    
    for epoch in range(epoch):
       # Compute and display the cost every 10 epoch
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("Current cost = %s" % cost(predictions, targets))
        # Init gragients
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0.
        # Go through each row
        for feature, target in zip(features, targets):
            # Compute prediction
            z = pre_activation(feature, weights, bias)
            y = activation(z)
            # Update gradients
            weights_gradients += (y - target) * d_activation(z) * feature
            bias_gradient += (y - target) * d_activation(z)
        # Update variables
        weights = weights - (learning_r * weights_gradients)
        bias = bias - (learning_r * bias_gradient)
    #print current accurency
    predictions = predict(features, weights, bias)
    print("Accurency: ", np.mean(predictions == targets))
if __name__ == "__main__":
    #rows
    features, targets = get_dataset()
    #variable
    weights, bias = init_variables()
    train(features, targets, weights, bias)

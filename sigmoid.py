import math

# correct :D
def sigmoid(Z):
    """Calculates the sigmoid for use in forward propagation."""
    t1 = 1/(1+math.e**-Z)
    return t1
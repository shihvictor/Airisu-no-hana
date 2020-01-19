import numpy as np

def randInitializeWeights(layerIn, layerOut):
    print("Initializing weights.")
    
    # scales random values to range  (-epsilon_init, epsilon_init). 
    epsilon_init = 0.12
    thetaInit = np.random.rand(layerOut, layerIn + 1) * epsilon_init * 2 - epsilon_init
    return thetaInit
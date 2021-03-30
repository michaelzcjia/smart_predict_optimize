# imports
import numpy as np

def generate_data(n, p, grid_dim, sigma, noise, degree):
    '''
    Generate data for nxn grid

    Parameters:
        int n: number of data points to generate
        int p: number of features
        int grid_dim: Dimension of square grid, determines size of cost vector
        array sigma: array of length p, is the variance of each feature vector dimension, i.e. x_i ~ N(0, sigma_p)
        float noise: multiplicative noise term applied to cost vector, sampled from uniform distribution in [1-noise, 1+noise]
        int degree: polynomial degree of generated cost vector. When degree=1, expected value of c is linear in x. Degree > 1 controls the amount of model misspecification.

    Returns:
        np.array X: feature data of dimension [num_samples, p]
        np.array C: cost data of dimension [num_samples, d]
    '''
    # Define number of edges based on griworksd size, i.e. size of cost vector
    d = grid_dim*(grid_dim-1)*2

    # Define the parameters of the true model
    B_star = np.random.binomial(size=[d,p], n=1, p= 0.5) # each entry of B is a bernoulli RV with prob = 0.5 entry is 1

    # Generate feature data: Generated from multivariate Gaussian distribution with i.i.d. standard normal entries --> x ~ N(0, sigma)
    X = np.random.normal(loc = 0, scale = sigma, size = [n, p]) # each row is a training point of size p

    # Generate cost data
    noise_vector = np.random.uniform(low = 1-noise, high = 1+noise, size = [n, d]) # i.i.d noise terms
    C = np.multiply((((1/np.sqrt(p) * B_star@X.T) + 3)**degree + 1).T, noise_vector)
    
    return X, C

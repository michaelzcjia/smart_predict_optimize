from HelperFunctions import *
import time
import pandas as pd
import pickle

def problem_size_experiment(params, noise, degree,sigma, iterations=30):
    
    ''' 
    Runs the direct and SGD solvers with given input parameters
    
    input: 
        dict{str:list} params: dictionary of parameter values to experiment with. Must specify 'n', 'p', and 'grid_size'
        float noise: multiplicative noise term applied to cost vector, sampled from uniform distribution in [1-noise, 1+noise]
        int degree: polynomial degree of generated cost vector. When degree=1, expected value of c is linear in x. Degree > 1 controls the amount of model misspecification.

    returns: dict{str:list} with experimental results including: runtime, SPO loss, and SPO plus loss for both direct and SGD solvers
    ''' 
    
    # Variable definitions
    experimental_results = {}
      
    # For each parameter combo solve the problem instance and record results
    for grid_dim in params['grid_dim']:
        for p in params['p']:
            for n in params['n']:
                # create sigma of length p
                sigma_arr = np.full(p,sigma)
                
                direct_runtimeparams = []
                SGD_runtimeparams = []
                
                SPO_loss_directparams = []
                SPO_loss_SGDparams = []
                
                SPO_plus_loss_directparams= []
                SPO_plus_loss_SGDparams = []
                # Create shortest path contraints
                A,b = CreateShortestPathConstraints(grid_dim)
                for i in range(iterations):
                    print(n,p,grid_dim,i)
                    # Generate the dataset
                    X, C = generate_data(n, p, grid_dim, sigma_arr, noise, degree)

                    #print('for n =', n, 'p = ', p, 'grid_dim = ',grid_dim)
                    # Run the direct solution and record the time
                    start_direct = time.time()
                    B_direct=DirectSolution(A,b, X, C)
                    end_direct = time.time() - start_direct
                    direct_runtimeparams.append(end_direct)

                    # Run the SGD solution and record the time
                    start_sgd = time.time()
                    B_SGD=GradientDescentSolution(A,b, X, C, batch_size=10,epsilon = 0.001) 
                    end_sgd = time.time() - start_sgd
                    SGD_runtimeparams.append(end_sgd)

                    # Record losses
                    solver = ShortestPathSolver(A,b)
                    SPO_loss_directparams.append(SPOLoss(solver, X, C, B_direct))
                    SPO_loss_SGDparams.append(SPOLoss(solver, X, C, B_SGD))
                    SPO_plus_loss_directparams.append(SPOplusLoss(solver, X, C, B_direct))
                    SPO_plus_loss_SGDparams.append(SPOplusLoss(solver, X, C, B_SGD))

                #store results from all iterations in dicts
                experimental_results[(n, p, grid_dim,'direct_runtime')] = direct_runtimeparams
                experimental_results[(n, p, grid_dim,'SGD_runtime')] = SGD_runtimeparams
                
                experimental_results[(n, p, grid_dim,'SPO_loss_direct')] = SPO_loss_directparams
                experimental_results[(n, p, grid_dim,'SPO_loss_SGD')] = SPO_loss_SGDparams
                
                experimental_results[(n, p, grid_dim,'SPO_plus_loss_direct')]= SPO_plus_loss_directparams
                experimental_results[(n, p, grid_dim,'SPO_plus_loss_SGD')] = SPO_plus_loss_SGDparams
            checkpoint = pd.DataFrame(experimental_results).transpose()
            checkpoint.index.names = ['n','p','grid_dim','metric']
            pickle.dump(checkpoint,open('SPOresultsCheckpoint.pkl','wb'))
    experimental_results = pd.DataFrame(experimental_results).transpose()
    experimental_results.index.names = ['n','p','grid_dim','metric']
    return experimental_results


params = {"n": [100,200,300,400,500,600,700,800,900,1000], "p": [5,10,15,20], "grid_dim": [5]}
noise = 0.25
degree = 3
sigma=0.2

experiment1 = problem_size_experiment(params, noise, degree,sigma,iterations=30)
pickle.dump(experiment1, open('SPOresultsAllVars.pkl','wb'))

import numpy as np
import cvxpy as cp


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


def CreateShortestPathConstraints(gridsize):
    '''
    Generate constraints for the nxn grid shortest path problem. 
    Each node in the grid has a constraint where the LHS is the inflows - outflows and the RHS is the desired flow.
    The desired flow is 0 for all nodes except for the start node where it's -1 and end node where it's 1
    
    Parameters:
        int gridsize: Size of each dimension in grid
        
    Returns:
        np.array A: Flow matrix of shape [num_nodes, num_edges]. Aij is -1 if the edge j is an outflow of node i and 1 if edge edge j is an inflow of node i
        np.array b: RHS of constraints [num_nodes]
    '''
    #define node and edge sizes
    num_nodes = gridsize**2
    num_directional_edges = (num_nodes - gridsize) #num vertical edges and num horizontal edges
    num_edges = num_directional_edges*2 #sum vertical and horizontal edges together
    
    #initialize empty A and B arrays
    A = np.zeros((num_nodes,num_edges),np.int8)
    b = np.zeros(num_nodes,np.int8)
    
    #Fill in flow matrix
    #nodes are ordered by rows. ex. in a 3x3 grid the first rows nodes are indices 1,2,3 and second row is 4,5,6
    #horizontal edges are enumerated first and then vertical edges
    horizontaledgepointer = 0
    verticaledgepointer = 0
    for i in range(num_directional_edges):
        #update flow matrix for horizontal edges
        outnode = horizontaledgepointer
        innode = horizontaledgepointer+1
        
        A[outnode,i] = -1
        A[innode,i] = 1
        horizontaledgepointer+=1
        if (horizontaledgepointer+1)%gridsize==0:#node is at right edge of the grid so  go to next row
            horizontaledgepointer+=1
        
        #update flow matrix for vertical edges
        outnode = verticaledgepointer
        innode = verticaledgepointer+gridsize
        A[outnode,num_directional_edges+i] = -1
        A[innode,num_directional_edges+i] = 1
        verticaledgepointer+=gridsize
        if verticaledgepointer+gridsize>=num_nodes:#node is at bottom edge of the grid so go to next column
            verticaledgepointer= (verticaledgepointer%gridsize)+1
        
    #update RHS for start and end nodes
    b[0] = -1
    b[-1] = 1 
    return A,b


class ShortestPathSolver:
    def __init__(self,A,b):
        '''
        Defines binary optimization problem to solve the shortest path problem with constraint matrix A and RHS b as numpy arrays
        Parameters:
            np.array A: constraint matrix A
            np.array B: RHS of constraints

        '''
        if A.shape[0]!=b.size:
            print('invalid input')
            return
        numedges = A.shape[1]
        self.c = cp.Parameter(numedges, nonneg=True)
        self.w = cp.Variable(numedges, boolean=True)
        self.prob = cp.Problem(cp.Minimize(self.c@self.w), 
                               [A @ self.w == b, A[0,:]@ self.w <= b[0]]) #add a trivial inequality constraint because necessary for GLPK_MI solver
        
    def solve(self,c):
        '''
        Solves the predefined optmiization problem with cost vector c and returns the decision variable array
        '''
        self.c.project_and_assign(c)
        self.prob.solve(solver='GLPK_MI')
        return self.w.value


def DirectSolution(A, b, X, C):
    '''
    Computes the direct solution that minimizes the SPO+ loss given the hypothesis class of linear models B
    
    Parameters:
        np.array A: Constraint matrix [num_nodes, num_edges]
        np.array b: RHS of constraints [num_nodes]
        np.array X: Feature Matrix [num_samples, num_features]
        np.array C: Cost Matrix [num_samples, num_edges]

    Returns:
        np.array B: coefficient matrix of fitted linear models [num_edges, num_features]
    '''
    num_samples=X.shape[0]

    #solve every shortest path problem
    solver = ShortestPathSolver(A,b)
    W=np.apply_along_axis(solver.solve,1,C)#W has shape [num_samples, num_edges]

    #define linear program variables
    B=cp.Variable((A.shape[1],X.shape[1])) #B has shape [num_edges, num_features]
    P=cp.Variable((num_samples,A.shape[0]), nonneg=True) #B has shape [num_samples, num_nodes]
    
    #define linear program objective and constraints
    objective = (cp.sum(-P@b) + 2*cp.sum(cp.multiply(X@B.T,W)) - cp.sum(cp.multiply(W,C)))/num_samples
    prob = cp.Problem(cp.Minimize(objective), 
                                   [(P@A) <= ((2*(X@B.T)) - C)])
    #solve
    prob.solve()
    return B.value

def GradientDescentSolution(A, b, X, C, batch_size=5, r=None, r_strength=0.1):
    '''
    Computes the direct solution that minimizes the SPO+ loss given the hypothesis class of linear models B
    
    Parameters:
        np.array A: Constraint matrix [num_nodes, num_edges]
        np.array b: RHS of constraints [num_nodes]
        np.array X: Feature Matrix [num_samples, num_features]
        np.array C: Cost Matrix [num_samples, num_edges]
        integer batch_size: batch size  

    Returns:
        np.array B: coefficient matrix of fitted linear models [num_edges, num_features]
    '''
    epsilon = 0.001
    loop = True 

    #solve every shortest path problem
    solver = ShortestPathSolver(A,b)
    W_c = np.apply_along_axis(solver.solve,1,C)#W has shape [num_samples, num_edges]
    B = np.zeros((A.shape[1],X.shape[1])) #B has shape [num_edges, num_features]
    
    step=0
    while loop:      
        # get a random sample of indices of size batch_size
        batch_indices = np.random.randint(0,len(X),batch_size)
        X_sample = X[batch_indices]
        C_sample = C[batch_indices]
        W_c_sample = W_c[batch_indices]
        
#         print(f'B: {B.shape}')
#         print(f'X: {X.shape}')
#         print(f'w_j_t: {objectives.shape}')
        
        # solve for the gradient of the unregularized objective function
        objectives = 2*(B@X_sample.T).T - C_sample
        W_batch=np.apply_along_axis(solver.solve,1,objectives)
        G_batch = (W_c_sample-W_batch).T@X_sample # might not be the same as mean
        
#         print(f'gradient: {G_batch.shape}')
        
        # calculate the gradient step  
        grad = G_batch/batch_size
        # with l2 regularization
        if r == "l2":
            grad_of_l2 = grad # TODO, derivative of frob norm
            learning_rate = 2/(r_strength(step+1))
            grad_step = learning_rate(grad + r_strength*grad)
        # without regularization 
        else:
            learning_rate = 1/(step+1)**(1/2)
            grad_step = learning_rate*grad
            
        # calculate new weights
        B_new = B + grad_step
        
        # stopping condition
        if np.mean(np.abs(B@X.T - B_new@X.T)) < epsilon: 
            loop = False 
            print(f'Converged after {step} steps')

        # update weights 
        B = B_new
        step += 1
            
    return B 

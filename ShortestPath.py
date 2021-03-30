import numpy as np
import cvxpy as cp

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


class Solver:
    def __init__(self,A,b):
        '''
        Constraint matrix A and RHS b as numpy arrays
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
        self.c.project_and_assign(c)
        self.prob.solve(solver='GLPK_MI')
        return self.w.value#, self.prob.value

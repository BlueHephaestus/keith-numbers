"""
NOTE: The methods in this file regarding integer linear programming problems
    are meant for obtaining keith numbers. They are not 
    general methods for generating and solving ILPs, so do not use them
    as such. 

-Blake Edwards / Dark Element
"""
import sys, os
import numpy as np
import time
from ortools.linear_solver import pywraplp
from joblib import Parallel, delayed
import multiprocessing

class suppress_stderr(object):
    """
    A context manager for doing a "deep suppression" of stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    Credit goes to: https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
        They're a lifesaver.
    """
    def __init__(self):
        # Open a null file
        self.null_fd =  os.open(os.devnull,os.O_RDWR)
        # Save the actual stderr (2) file descriptor.
        self.save_fd = os.dup(2)

    def __enter__(self):
        # Assign the null pointer to stderr.
        os.dup2(self.null_fd,2)

    def __exit__(self, *_):
        # Re-assign the real stderr back to (2)
        os.dup2(self.save_fd,2)
        # Close file descriptor
        os.close(self.null_fd)
        os.close(self.save_fd)

def get_ilp(d):
    """
    Get a matrix where each row is a set of coefficients
        for this number of digits, and can be solved to obtain
        a keith number.
    """

    #Create repfib matrix
    m = np.zeros((d,d))
    m[-1,:] = np.ones((d,))
    m[:-1, 1:] = np.eye(d-1)

    i = d-1
    m = list(m)#So we can append easily
    while 9 * np.sum(m[i]) < 10**(d-1):
        i+=1
        #Append sum of rows i-d:i to m
        m.append(np.sum(m[i-d:i], axis=0))

    #Create final coefficient matrix from this
    c = []
    while m[i][0] <= 10**(d-1):
        c.append(m[-1])
        i+=1
        m.append(np.sum(m[i-d:i], axis=0))

    c -= 10**np.arange(d-1, -1, -1)

    return c


def solve_ilp(c):
    """
    Assumes len(c) == d, where d is the number of digits.

    As a result, our solution vector x will also have this length,
        so that len(c) == len(x) == d.

    Solve an integer linear programming problem constrained with the given constant coefficients
        s.t.  c_0*x_0 + c_1*x_1 + ... c_(d-1)*x_(d-1) = 0 

    For this number of digits d. 

    Additionally, we constrain all solutions x to be 0 <= x <= 9
        and constrain the first input variable s.t. x_0 >= 1.

    We return the integer solution vector x.
    """
    d = len(c)#Get number of digits

    solver = pywraplp.Solver('ILP Solver', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    x = [solver.IntVar(1.0, 9.0, 'x_0')] 
    for i in range(d-1):
        x.append(solver.IntVar(0.0, 9.0, 'x_{}'.format(i+1)))

    # c_0*x_0 + c_1*x_1 + ... = 0
    constraint = solver.Constraint(0,0)
    for i in range(d):
        constraint.SetCoefficient(x[i], c[i])

    # Minimize x
    objective = solver.Objective()
    for i in range(d):
        objective.SetCoefficient(x[i], 1)
    objective.SetMinimization()

    #Solve the ILP with our constraints.
    with suppress_stderr():
        solver.Solve()

    #Return an int solution vector
    with suppress_stderr():
        #Suppress since we know it will just return a value of 0 if the optimization fails
        res = [int(var.solution_value()) for var in x]
    return res

def is_keith_number(n):
    #Check if this is a keith number. 

    if n == 0: return False

    x = [int(d) for d in str(n)]
    y = sum(x)

    while y < n:
        #pop an element, push an element
        x, y = x[1:]+[y], 2*y-x[0]

    #If it's equal now, then it's by definition a keith number
    return y == n

d = int(sys.argv[1])#number of digits

#Solve all ILPs asynchronously
core_n = multiprocessing.cpu_count()
ns = Parallel(n_jobs=core_n)(delayed(solve_ilp)(c) for c in get_ilp(d))

#Print keith numbers found
for n in ns:
    n = int("".join(map(str, n)))#Number obtained
    if is_keith_number(n):
        print(n)


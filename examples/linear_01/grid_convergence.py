import sys
sys.path.append("../../src")

import numpy as np
from equation import ConvexEquation
from solver import Solver
from matplotlib.pyplot import*

if __name__ == "__main__":
    eqn = ConvexEquation()
    eqn.setup()
    print("Energy Function: ", eqn.w)
    print("Differentiated Equation: ", eqn.f)
    nrange = np.array(range(1,4))
    error = []
    for n in nrange:
        # define domain, note complex type
        x = np.linspace(0.0, 1.0, 10**n, dtype=np.complex)
        # intialize solution
        u = np.zeros_like(x, dtype=x.dtype)
        # solve
        solver = Solver(eqn, x, u)
        solver.maxiter = 5
        solver.dt = 1e6
        solver.run()
        # get analytic solution
        u_analytic = eqn.analytic_solution(x)
        e_ = np.sqrt(sum((u_analytic-u)**2))*(x[1] - x[0])
        error.append(e_)
        
    # plot
    loglog(10**nrange, error, 'ro--')
    xlabel('N')
    ylabel('error')
    legend()
    show()

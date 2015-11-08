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
    
    # define domain, note complex type
    x = np.linspace(0.0, 1.0, 200, dtype=np.complex)

    # intialize solution
    u = np.zeros_like(x, dtype=x.dtype)

    # solve
    solver = Solver(eqn, x, u)
    solver.maxiter = 3
    solver.dt = 1e10
    solver.run()
    # get analytic solution
    u_analytic = eqn.analytic_solution(x)
    
    # plot
    plot(x, u, 'o',label='Numerical')
    plot(x, u_analytic, 'r--', label='Analytic')
    xlabel('x')
    ylabel('u(x)')
    legend(loc=2)
    savefig("solution.pdf")
    show()

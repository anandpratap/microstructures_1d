import sys
sys.path.append("../../src")

import numpy as np
from equation import NonConvexEquation
from solver import Solver
from matplotlib.pyplot import*

if __name__ == "__main__":
    eqn = NonConvexEquation()
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
     # plot
    figure()
    subplot(211)
    plot(x, u, 'r-')
    xlabel('x')
    ylabel('u')
    gca().ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    subplot(212)
    plot(x, solver.u_x, 'r-')
    xlabel('x')
    ylabel('u\'(x)')
    gca().ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    show()
    savefig("solution.pdf")

import sys
sys.path.append("../../src")

import numpy as np

from equation import ConvexEquationParam
from solver import Solver, AdjointSolver
from matplotlib.pyplot import*

if __name__ == "__main__":
    eqn = ConvexEquationParam()
    eqn.setup()
    print("Energy Function: ", eqn.w)
    print("Differentiated Equation: ", eqn.f)
    obj = lambda u : sum(u*u)
    x = np.linspace(0.0, 1.0, 100, dtype=np.complex)
    u = np.zeros_like(x, dtype=x.dtype)
    solver = Solver(eqn, x, u)
    solver.maxiter = 3
    solver.dt = 1e10
    solver.param = 1.0
    solver.run()
    np.savetxt("u_benchmark", solver.u.astype(np.float64) + np.random.randn(100)*0.0001)

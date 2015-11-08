import sys
sys.path.append("../../src")

import numpy as np
from equation import NonConvexEquationParam
from solver import Solver, AdjointSolver
from matplotlib.pyplot import*

if __name__ == "__main__":
    eqn = NonConvexEquationParam()
    eqn.setup()
    print("Energy Function: ", eqn.w)
    print("Differentiated Equation: ", eqn.f)
    obj = lambda u : sum(u*u)
    x = np.linspace(0.0, 1.0, 1000, dtype=np.complex)
    u = np.zeros_like(x, dtype=x.dtype)
    solver = Solver(eqn, x, u)
    solver.maxiter = 3
    solver.dt = 1e10
    solver.run()

    adsolver = AdjointSolver(eqn, x, u, obj) 
    sens = adsolver.sens
    obj_base = obj(u)

    fd = []
    dparams = 1.0/10.0**np.array(range(1, 14))
    for dparam in dparams:
        x = np.linspace(0.0, 1.0, 1000, dtype=np.complex)
        u = np.zeros_like(x, dtype=x.dtype)
        solver = Solver(eqn, x, u)
        solver.maxiter = 3
        solver.dt = 1e10
        solver.param = solver.param + dparam
        solver.run()
        obj_p = obj(u)
        fd.append((obj_p - obj_base)/dparam)
        
    figure()
    loglog(dparams, abs(np.ones_like(dparams)*sens-np.array(fd)), 'r--', label="Adjoints - Finite Diff")
    xlabel("dparam")
    ylabel("dJ/dparam")
    legend()
    savefig("sensitivity.pdf")
    show()

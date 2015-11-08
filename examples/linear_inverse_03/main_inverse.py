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
    
    ub = np.loadtxt("u_benchmark")
    obj = lambda u : sum((u-ub)*(u-ub))
    param = 0.1
    stepsize = 0.1
    for i in range(10):
        x = np.linspace(0.0, 1.0, 100, dtype=np.complex)
        u = np.zeros_like(x, dtype=x.dtype)
        solver = Solver(eqn, x, u)
        solver.maxiter = 2
        solver.dt = 1e14
        solver.param = param
        solver.run()
        adsolver = AdjointSolver(eqn, x, u, obj) 
        sens = adsolver.sens
        obj_base = obj(u)
        param = param - sens/abs(sens)*stepsize
        if i == 0:
            u_prior = solver.u
        print("Inverse Step ", i, "OBJ: ", obj_base)
        

    figure()
    plot(x, u_prior, "g-", label="Prior")
    plot(x, solver.u, "r-", label="Posterior")
    plot(x[::5], ub[::5], "b.", label="Benchmark")
    legend()
    show()

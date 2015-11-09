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
    obj = lambda u, param : sum((u-ub)*(u-ub))/(1e-5)**2 + param/1.0**2
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
        adsolver = AdjointSolver(eqn, x, solver.u, obj) 
        sens = adsolver.sens
        obj_base = obj(solver.u, solver.param)
        param = param - sens/abs(sens)*stepsize
        if i == 0:
            u_prior = np.copy(solver.u)
        print("Inverse Step ", i, "OBJ: ", obj_base)
        

    u__ = np.copy(solver.u)
    figure(1)
    plot(x, u_prior, "g-", label="Prior")
    plot(x, solver.u, "r-", label="Posterior")
    plot(x[::5], ub[::5], "b.", label="Benchmark")

    ub = np.loadtxt("u_benchmark")
    ub += np.random.randn(np.size(ub))*1e-3
    obj = lambda u, param : sum((u-ub)*(u-ub))/(1e-3)**2 + param/1.0**2
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
        adsolver = AdjointSolver(eqn, x, solver.u, obj) 
        sens = adsolver.sens
        obj_base = obj(solver.u, solver.param)
        param = param - sens/abs(sens)*stepsize
        print("Inverse Step ", i, "OBJ: ", obj_base)
        
    figure(1)
    plot(x, solver.u, "k--", label="Posterior With noise")
    plot(x[::5], ub[::5], "b.-", label="Benchmark with noise")
    legend()
    xlabel('x')
    ylabel('u')
    savefig("inverse.pdf")
    
    figure(2)
    plot(x, solver.u-u__, "k--", label="Diff w and w/o noise")
    legend()
    xlabel('x')
    ylabel('u diff')
    savefig("diff.pdf")
    show()

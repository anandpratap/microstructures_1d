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
        

    u__ = np.copy(u)
    figure(1)
    plot(x, u_prior, "g-", label="Prior")
    plot(x, u, "r-", label="Posterior")
    plot(x[::5], ub[::5], "b.", label="Benchmark")

    ub_ = np.loadtxt("u_benchmark")
    ub_ += np.random.randn(np.size(ub_))*1e-3
    obj_ = lambda u : sum((u-ub_)*(u-ub_))
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
        adsolver = AdjointSolver(eqn, x, u, obj_) 
        sens = adsolver.sens
        obj_base = obj_(u)
        param = param - sens/abs(sens)*stepsize
        print("Inverse Step ", i, "OBJ: ", obj_base)
        
    figure(1)
    plot(x, u, "k--", label="Posterior With noise")
    plot(x[::5], ub_[::5], "b.-", label="Benchmark with noise")
    legend()
    xlabel('x')
    ylabel('u')
    savefig("inverse.pdf")
    
    figure(2)
    plot(x, u-u__, "k--", label="Diff w and w/o noise")
    legend()
    xlabel('x')
    ylabel('u diff')
    savefig("diff.pdf")
    show()

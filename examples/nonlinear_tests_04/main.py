import sys
sys.path.append("../../src")

import numpy as np
from equation import NonConvexEquationParam
from solver import Solver
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    eqn = NonConvexEquationParam()
    eqn.setup()
    print("Energy Function: ", eqn.w)
    print("Differentiated Equation: ", eqn.f)
    
    # define domain, note complex type
    x = np.linspace(0.0, 1.0, 200, dtype=np.complex)
    param_ = np.linspace(0.1, 1.0, 1000, dtype=x.dtype)
    paramm, xx = np.meshgrid(param_, x)
    uu = np.zeros_like(paramm)
    for idx, param in enumerate(param_):
        # intialize solution
        u = np.zeros_like(x, dtype=x.dtype)
        # solve
        solver = Solver(eqn, x, u)
        solver.maxiter = 3
        solver.dt = 1e10
        solver.param = param
        solver.run()
        # plot
        uu[:,idx] = solver.u[:]
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(paramm*1.0/8.0, xx, uu, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.savefig('surf_u.pdf')

    plt.figure()
    u_max = np.amax(uu, axis=0)
    plt.semilogy(param_*1.0/8.0, u_max, 'r-')
    plt.xlabel('l')
    plt.ylabel('max (u)')
    plt.savefig('max_u.pdf')
    plt.show()
    

import numpy as np
import sympy as sp
import scipy.sparse as sparse
from sympy.utilities.autowrap import ufuncify
from scipy.sparse.linalg import spsolve

class Solver(object):
    def __init__(self, equation, x, u, verbose=False):
        self.equation = equation
        self.x = np.copy(x)
        self.u = np.copy(u)
        self.n = np.size(self.u)
        self.dx = x[1] - x[0]
        assert(np.size(x) == np.size(u) == self.n)

        self.u_x = np.zeros(self.n, dtype=self.x.dtype)
        self.u_xx = np.zeros(self.n, dtype=self.x.dtype)
        self.u_xxx = np.zeros(self.n, dtype=self.x.dtype)
        self.u_xxxx = np.zeros(self.n, dtype=self.x.dtype)

        self.maxiter = 10
        self.dt = 1e7
        self.parse_equation()
        self.param = 1.0
        self.verbose = verbose
        self.tol = 1e-10
        self.hbc = True

    def calc_first_der(self, u):
        self.u_x[1:-1] = (u[2:] - u[0:-2])/(2*self.dx)
    
    def calc_second_der(self, u):
        self.u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[0:-2])/self.dx**2
        
    def calc_third_der(self, u):
        raise NotImplementedError("Third derivative is not available!")
        
    def calc_fourth_der(self, u):
        self.u_xxxx[2:-2] = (u[0:-4] - 4.0*u[1:-3] + 6.0*u[2:-2] - 4.0*u[3:-1] + u[4:])/self.dx**4
        if self.hbc:
            ub = 4.0*(-5.0/6.0*u[0] + 1.5*u[1] - 0.5*u[2] + 1.0/12.0*u[3])
        else:
            ub = u[1]
        self.u_xxxx[1] = (ub - 4.0*u[0] + 6.0*u[1] - 4.0*u[2] + u[3])/self.dx**4
        if self.hbc:
            ub = -4.0*(5.0/6.0*u[-1] - 1.5*u[-2] + 0.5*u[-3] - 1.0/12.0*u[-4])
        else:
            ub = u[-2]
        self.u_xxxx[-2] = (u[-4] - 4.0*u[-3] + 6.0*u[-2] - 4.0*u[-1] + ub)/self.dx**4

    def parse_equation(self):
        if hasattr(self.equation, "param"):
            args = (self.equation.eps, self.equation.u_xx, self.equation.u_xxxx, self.equation.param)
        else:
            args = (self.equation.eps, self.equation.u_xx, self.equation.u_xxxx)
        self.f = sp.lambdify(args, self.equation.f, "numpy")
    def calc_residual(self, u):
        if hasattr(self.equation, "param"):
            R = self.equation.boundary(self.x, u, self.param)
        else:
            R = self.equation.boundary(self.x, u)

        __funcs__ = [self.calc_first_der, self.calc_second_der, self.calc_fourth_der]
        for func in __funcs__:
            func(u)
        if hasattr(self.equation, "param"):
            R_ = self.f(self.u_x, self.u_xx, self.u_xxxx, self.param)
        else:
            R_ = self.f(self.u_x, self.u_xx, self.u_xxxx)
        R[1:-1] = R_[1:-1]
        assert(np.size(R) == self.n)
        return R

    def calc_residual_jac(self, u):
        dRdU = np.zeros([self.n, self.n], dtype=u.dtype)
        du = 1e-24
        for i in range(self.n):
            u[i] = u[i] + 1j*du
            R = self.calc_residual(u)
            u[i] = u[i] - 1j*du
            dRdU[:,i] = np.imag(R[:])/du
        dRdU = sparse.csr_matrix(dRdU)
        return dRdU

    def step(self):
        R = self.calc_residual(self.u)
        dRdU = self.calc_residual_jac(self.u)
        A = sparse.eye(self.u.size, dtype=self.u.dtype)/self.dt - dRdU
        du = spsolve(A.astype(np.float64), R.astype(np.float64))
        self.u[:] = self.u[:] + du[:]
        return np.linalg.norm(du)

    def run(self):
        for iter in range(self.maxiter):
            norm = self.step()
            if self.verbose:
                print("Iteration number: " , iter, " dU Norm: ", norm)
            if norm < self.tol:
                break
    def setup(self):
        pass


class AdjointSolver(Solver):
    def __init__(self, equation, x, u, obj, param):
        Solver.__init__(self, equation, x, u)
        self.param = param
        self.obj = obj
        self.sens = self.calc_sensitivity()
    def calc_dJdU(self, u):
        dJdU = np.zeros(self.n, dtype=u.dtype)
        du = 1e-24
        for i in range(self.n):
            u[i] = u[i] + 1j*du
            dJdU[i] = np.imag(self.obj(u, self.param))/du
            u[i] = u[i] - 1j*du
        return dJdU

    def calc_psi(self):
        dJdU = self.calc_dJdU(self.u)
        dRdU = self.calc_residual_jac(self.u)
        psi = spsolve(dRdU.T, -dJdU.T)
        return psi
        
    def calc_residual_dparam(self, u):
        dparam = 1e-24
        if hasattr(self.param, "__len__"):
            dRdparam = np.zeros([self.n, self.n], dtype=u.dtype)
            for i in range(self.n):
                self.param[i] = self.param[i] + 1j*dparam
                R = self.calc_residual(u)
                self.param[i] = self.param[i] - 1j*dparam
                dRdparam[:,i] = np.imag(R[:])/dparam
        else:
            self.param = self.param + 1j*dparam
            R = self.calc_residual(u)
            self.param = self.param - 1j*dparam
            dRdparam = np.imag(R[:])/dparam
        return dRdparam

    def calc_obj_param(self):
        dparam = 1e-24
        self.param = self.param + 1j*dparam
        delJ = np.imag(self.obj(self.u, self.param))/dparam
        self.param = self.param - 1j*dparam
        return delJ

    def calc_sensitivity(self):
        dR = self.calc_residual_dparam(self.u)
        psi = self.calc_psi()
        delJ = self.calc_obj_param()
        sens = delJ + psi.T.dot(dR)
        return sens

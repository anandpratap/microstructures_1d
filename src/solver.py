import numpy as np
import sympy as sp
import scipy.sparse as sparse
from sympy.utilities.autowrap import ufuncify
from scipy.sparse.linalg import spsolve
from itertools import combinations

class Solver(object):
    def __init__(self, equation, x, u, verbose=False):
        self.equation = equation
        self.x = np.copy(x)
        self.u = np.copy(u)
        self.n = np.size(self.u)
        assert(np.size(x) == np.size(u) == self.n)

        self.u_x = np.zeros(self.n, dtype=self.x.dtype)
        self.u_xx = np.zeros(self.n, dtype=self.x.dtype)
        self.u_xxx = np.zeros(self.n, dtype=self.x.dtype)
        self.u_xxxx = np.zeros(self.n, dtype=self.x.dtype)

        self.maxiter = 10
        self.dt = 1e7
        self.parse_equation()
        self.param = 0.0
        self.verbose = verbose
        self.tol = 1e-12
        self.hbc = True

    def calc_first_der(self, u):
        x = self.x
        self.u_x[1:-1] = (u[2:] - u[0:-2])/(x[2:] - x[0:-2])
        self.u_x[0] = (u[0] - u[1])/(x[0] - x[1])
        self.u_x[-1] = (u[-1] - u[-2])/(x[-1] - x[-2])
        return self.u_x
    
    def calc_second_der(self, u):
        deta = 1.0
        x = self.x
        nx = np.size(x)
        
        dx_1 = x[1:-1] - x[0:-2]
        dx_2 = x[2:] - x[1:-1]
        
        u_xx = np.zeros(nx, dtype = np.complex)
        u_xx[1:-1] = 2.0*(dx_1*u[2:] - (dx_1 + dx_2)*u[1:-1] + dx_2*u[0:-2])/(dx_1*dx_2*(dx_1 + dx_2))
        self.u_xx[:] = u_xx[:]
        return self.u_xx
        
    def calc_third_der(self, u):
        deta = 1.0
        x = self.x
        nx = np.size(x)
        u_xxx = np.zeros(nx, dtype = np.complex)
        alpha = np.zeros([5, nx-4], dtype = np.complex)
        alpha[0,:] = x[0:-4] - x[2:-2]
        alpha[1,:] = x[1:-3] - x[2:-2]
        alpha[2,:] = x[2:-2] - x[2:-2]
        alpha[3,:] = x[3:-1] - x[2:-2]
        alpha[4,:] = x[4:] - x[2:-2]
        num = np.zeros([5, nx-4], dtype=x.dtype)
        den = np.ones([5, nx-4], dtype=x.dtype)
        
        for i in range(5):
            for j in range(5):
                if i != j:
                    num[i] += alpha[j,:]
                    den[i] *= (alpha[i,:] - alpha[j,:])
        num = -6.0*num
        fac = num/den
        u_xxx[2:-2] = fac[0,:]*u[0:-4] + fac[1,:]*u[1:-3] + fac[2,:]*u[2:-2] + fac[3,:]*u[3:-1] + fac[4,:]*u[4:]
        
        # index 1
        alpha = np.zeros(5, dtype=x.dtype)
        alpha[0] = x[0] - x[1]
        alpha[1] = x[1] - x[1]
        alpha[2] = x[2] - x[1]
        alpha[3] = x[3] - x[1]
        alpha[4] = x[4] - x[1]
        num = np.zeros(5, dtype=x.dtype)
        den = np.ones(5, dtype=x.dtype)
        
        for i in range(5):
            for j in range(5):
                if i != j:
                    num[i] += alpha[j]
                    den[i] *= (alpha[i] - alpha[j])
        num = -6.0*num
        fac = num/den
        for i in range(5):
            u_xxx[1] += fac[i]*u[i]
            
        alpha = np.zeros(5, dtype=x.dtype)
        alpha[0] = x[-5] - x[-2]
        alpha[1] = x[-4] - x[-2]
        alpha[2] = x[-3] - x[-2]
        alpha[3] = x[-2] - x[-2]
        alpha[4] = x[-1] - x[-2]
        num = np.zeros(5, dtype=x.dtype)
        den = np.ones(5, dtype=x.dtype)
        for i in range(5):
            for j in range(5):
                if i != j:
                    num[i] += alpha[j]
                    den[i] *= (alpha[i] - alpha[j])
        num = -6.0*num
        fac = num/den
        for i in range(5):
            u_xxx[-2] += fac[i]*u[-5+i]
        self.u_xxx[:] = u_xxx[:]
        return self.u_xxx
                
    def calc_fourth_der(self, u):
        deta = 1.0
        x = self.x
        nx = np.size(x)
        u_xxxx = np.zeros(nx, dtype = np.complex)
        
        alpha = np.zeros([5, nx-4], dtype = np.complex)
        alpha[0,:] = x[0:-4] - x[2:-2]
        alpha[1,:] = x[1:-3] - x[2:-2]
        alpha[2,:] = x[2:-2] - x[2:-2]
        alpha[3,:] = x[3:-1] - x[2:-2]
        alpha[4,:] = x[4:] - x[2:-2]
        den = np.ones([5, nx-4], dtype=x.dtype)
        
        for i in range(5):
            for j in range(5):
                if i != j:
                    den[i] *= (alpha[i,:] - alpha[j,:])
        fac = 24.0/den
        u_xxxx[2:-2] = fac[0,:]*u[0:-4] + fac[1,:]*u[1:-3] + fac[2,:]*u[2:-2] + fac[3,:]*u[3:-1] + fac[4,:]*u[4:]
        
        # index 1
        if self.hbc:
            alpha = np.zeros(5, dtype=x.dtype)
            xb = x[0]-(x[1] - x[0])
            alpha[0] = xb - x[0]
            alpha[1] = x[0] - x[0]
            alpha[2] = x[1] - x[0]
            alpha[3] = x[2] - x[0]
            alpha[4] = x[3] - x[0]
            den = np.ones(5, dtype=x.dtype)
            num = np.zeros(5, dtype=x.dtype)
            for i in range(5):
                for j in range(5):
                    if i != j:
                        den[i] *= (alpha[i] - alpha[j])
                tmp_list = range(5)
                del tmp_list[i]
                combs = list(combinations(tmp_list, 3))
                for k in combs:
                    num[i] += alpha[k[0]]*alpha[k[1]]*alpha[k[2]]
            fac = -num/den
            dudx = 0.0
            ub = (dudx - u[3]*fac[4] - u[2]*fac[3] - u[1]*fac[2] - u[0]*fac[1])/(fac[0] + 1e-16)
        else:
            ub = u[1]
        alpha = np.zeros(5, dtype=x.dtype)
        xb = x[0]-(x[1] - x[0])
        alpha[0] = xb - x[1]
        alpha[1] = x[0] - x[1]
        alpha[2] = x[1] - x[1]
        alpha[3] = x[2] - x[1]
        alpha[4] = x[3] - x[1]
        den = np.ones(5, dtype=x.dtype)
        for i in range(5):
            for j in range(5):
                if i != j:
                    den[i] *= (alpha[i] - alpha[j])
        fac = 24.0/den
        u_xxxx[1] = fac[0]*ub + fac[1]*u[0] + fac[2]*u[1] + fac[3]*u[2] + fac[4]*u[3]


        #index -2
        if self.hbc:
            alpha = np.zeros(5, dtype=x.dtype)
            xb = x[-1]+(x[-1] - x[-2])
            alpha[0] = xb - x[-1]
            alpha[1] = x[-1] - x[-1]
            alpha[2] = x[-2] - x[-1]
            alpha[3] = x[-3] - x[-1]
            alpha[4] = x[-4] - x[-1]
            den = np.ones(5, dtype=x.dtype)
            num = np.zeros(5, dtype=x.dtype)
            for i in range(5):
                for j in range(5):
                    if i != j:
                        den[i] *= (alpha[i] - alpha[j])
                tmp_list = range(5)
                del tmp_list[i]
                combs = list(combinations(tmp_list, 3))
                for k in combs:
                    num[i] += alpha[k[0]]*alpha[k[1]]*alpha[k[2]]
            fac = -num/den
            dudx = 0.0
            ub = (dudx - u[-4]*fac[4] - u[-3]*fac[3] - u[-2]*fac[2] - u[-1]*fac[1])/(fac[0] + 1e-16)
        else:
            ub = u[-2]
        
        alpha = np.zeros(5, dtype=x.dtype)
        xb = x[-1]+(x[-1] - x[-2])
        alpha[0] = xb - x[-2]
        alpha[1] = x[-1] - x[-2]
        alpha[2] = x[-2] - x[-2]
        alpha[3] = x[-3] - x[-2]
        alpha[4] = x[-4] - x[-2]
        den = np.ones(5, dtype=x.dtype)
        
        for i in range(5):
            for j in range(5):
                if i != j:
                    den[i] *= (alpha[i] - alpha[j])
        fac = 24.0/den
        u_xxxx[-2] = fac[0]*ub + fac[1]*u[-1] + fac[2]*u[-2] + fac[3]*u[-3] + fac[4]*u[-4]
        self.u_xxxx[1:-1] = u_xxxx[1:-1]
        return self.u_xxxx

    def parse_equation(self):
        if hasattr(self.equation, "param"):
            argsf = (self.equation.eps, self.equation.u_xx, self.equation.u_xxxx, self.equation.param)
            argsw = (self.equation.eps, self.equation.eps_x, self.equation.u_xx, self.equation.u_xxxx, self.equation.param)
        else:
            argsf = (self.equation.eps, self.equation.u_xx, self.equation.u_xxxx)
            argsw = (self.equation.eps, self.equation.eps_x, self.equation.u_xx, self.equation.u_xxxx)
        self.f = sp.lambdify(argsf, self.equation.f, "numpy")
        self.w = sp.lambdify(argsw, self.equation.w, "numpy")

    def calc_total_energy(self, u):
        __funcs__ = [self.calc_first_der, self.calc_second_der, self.calc_fourth_der, self.calc_third_der]
        for func in __funcs__:
            func(u)

        if hasattr(self.equation, "param"):
            w = self.w(self.u_x, self.u_xx, self.u_xx, self.u_xxxx, self.param)
        else:
            w = self.w(self.u_x, self.u_xx, self.u_xx, self.u_xxxx)
        W = 0.0
        for i in range(len(self.x)-1):
            dx = self.x[i+1] - self.x[i]
            w_avg = (w[i+1] + w[i])*0.5
            W += w_avg*dx
        return W

    def calc_residual(self, u):
        if hasattr(self.equation, "param"):
            R = self.equation.boundary(self.x, u, self.param)
        else:
            R = self.equation.boundary(self.x, u)

        __funcs__ = [self.calc_first_der, self.calc_second_der, self.calc_fourth_der, self.calc_third_der]
        for func in __funcs__:
            func(u)
        if hasattr(self.equation, "param"):
            R_ = self.f(self.u_x, self.u_xx, self.u_xxxx, self.param)
        else:
            R_ = self.f(self.u_x, self.u_xx, self.u_xxxx)

        source = self.param[4]*3072.0*self.u_xxxx
        R[1:-1] = R_[1:-1] + source[1:-1]
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

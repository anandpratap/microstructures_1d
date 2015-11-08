import numpy as np
import sympy as sp
from difftotal import difftotal

class Equation(object):
    def __init__(self):
        self.eps, self.eps_x = sp.symbols("eps eps_x")
        self.x, self.u_x, self.u_xx, self.u_xxx, self.u_xxxx = sp.symbols("x u_x u_xx u_xxx u_xxxx")

    def energy_function(self):
        raise NotImplementedError("Energy function not implemented!")
        
    def set_params(self):
        pass

    def setup(self):
        self.set_params()
        self.energy_function()
        self.get_sigma()
        self.get_beta()
        self.f = self.sigma_f - self.beta_f

    def get_sigma(self):
        self.sigma = sp.diff(self.w, self.eps)
        self.sigma_f = difftotal(self.sigma, self.x, {self.eps:self.u_x});
        self.sigma_f = difftotal(self.sigma_f, self.x, {self.u_x:self.u_xx});

    def get_beta(self):
        self.beta = sp.diff(self.w, self.eps_x)
        self.beta_f = difftotal(self.beta, self.x, {self.eps_x:self.u_xx});
        self.beta_f = difftotal(self.beta_f, self.x, {self.u_xx:self.u_xxx});
        self.beta_f = difftotal(self.beta_f, self.x, {self.u_xxx:self.u_xxxx});

    def boundary(self, x, u):
        raise NotImplementedError("Boundary condition not implemented!")
        
    def analytic_solution(self, x):
        raise NotImplementedError("Analytic solution not implemented!")


class ConvexEquation(Equation):
    def __init__(self):
        Equation.__init__(self)

    def energy_function(self):
        self.mu = 1.0
        self.l = 0.1
        self.t = self.g = 0.005
        self.w = 0.5*self.mu*self.eps**2 + 0.5*self.mu*self.l**2*self.eps_x**2

    def boundary(self, x, u):
        R = np.zeros(np.shape(u), dtype=u.dtype)
        dx = x[1] - x[0]
        R[0] = -u[0]
        uxxx = (2.5*u[-1] - 9.0*u[-2] + 12.0*u[-3] - 7.0*u[-4] + 1.5*u[-5])/dx**3
        R[-1] = (self.mu*self.l**2*uxxx + self.t)
        return R
        
    def analytic_solution(self, x):
        num = self.t*self.l*(1 - np.exp(1.0/self.l) + np.exp((1.0 - x)/self.l) - np.exp(x/self.l))
        den = self.mu*(np.exp(1.0/self.l) + 1)
        return num/den + self.t/self.mu*x

        
class ConvexEquationParam(Equation):
    def __init__(self):
        Equation.__init__(self)
        self.param = sp.Symbol("param")

    def energy_function(self):
        self.mu = 1.0
        self.l = 0.1*self.param
        self.t = self.g = 0.005
        self.w = 0.5*self.mu*self.eps**2 + 0.5*self.mu*self.l**2*self.eps_x**2

    def boundary(self, x, u, param):
        R = np.zeros(np.shape(u), dtype=u.dtype)
        dx = x[1] - x[0]
        R[0] = -u[0]
        uxxx = (2.5*u[-1] - 9.0*u[-2] + 12.0*u[-3] - 7.0*u[-4] + 1.5*u[-5])/dx**3
        R[-1] = (self.mu*self.l**2*uxxx + self.t).subs(self.param, param)
        return R
        
    def analytic_solution(self, x, param):
        l = np.float64(self.l.subs(self.param, param))
        num = self.t*l*(1 - np.exp(1.0/l) + np.exp((1.0 - x)/l) - np.exp(x/l))
        den = self.mu*(np.exp(1.0/l) + 1)
        rexpr = (num/den + self.t/self.mu*x)
        return rexpr
        

class NonConvexEquation(Equation):
    def __init__(self):
        Equation.__init__(self)
                
    def energy_function(self):
        self.mu = 1.0
        self.g = 2**(-12)
        self.l = 1/8.0
        alpha = 1/4.0
        self.w = self.mu*(self.eps**4 - 2*alpha**2*self.eps**2)/alpha**4 + 0.5*self.mu*self.l**2*self.eps_x**2

    def boundary(self, x, u):
        R = np.zeros(np.shape(u), dtype=u.dtype)
        R[0] = u[0]
        R[-1] = (u[-1] - self.g)
        return R


class NonConvexEquationParam(Equation):
    def __init__(self):
        Equation.__init__(self)
        self.param = sp.Symbol("param")
        
    def energy_function(self):
        self.mu = 1.0
        self.g = 2**(-12)
        self.l = 1/8.0*self.param
        alpha = 1/4.0
        self.w = self.mu*(self.eps**4 - 2*alpha**2*self.eps**2)/alpha**4 + 0.5*self.mu*self.l**2*self.eps_x**2

    def boundary(self, x, u, param):
        R = np.zeros(np.shape(u), dtype=u.dtype)
        R[0] = u[0]
        R[-1] = (u[-1] - self.g)
        return R

      


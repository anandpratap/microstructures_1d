import sys
sys.path.append("../../src")
import random
import numpy as np
from equation import NonConvexEquationParam
from solver import Solver
from matplotlib.pyplot import *
import matplotlib.mlab as mlab
import pymc3 as mc
import theano
import theano.tensor as T 
from theano.compile.ops import as_op
from scipy import stats
from scipy.stats.kde import gaussian_kde

if __name__ == "__main__":
    param_prior = 5.0
    sigma_obs = 1e-10
    sigma_prior = 0.1
    verbose = False
    u_benchmark = np.loadtxt("u_benchmark")
    x = np.linspace(0.0, 1.0, 201, dtype=np.complex)

    @as_op(itypes=[theano.tensor.dscalar], otypes=[theano.tensor.dvector])        
    def objective(param):
        eqn = NonConvexEquationParam()
        eqn.setup()
        u = np.zeros_like(x, dtype=x.dtype)
         # solve
        solver = Solver(eqn, x, u, verbose=verbose)
        solver.maxiter = 10
        solver.dt = 1e10
        solver.tol = 1e-6
        solver.param = complex(param)
        solver.run()
        u = solver.u.copy()
        if param < 0:
            u *= 1e10
        return u.astype(np.float64)


    # calculate prior solution
    eqn = NonConvexEquationParam()
    eqn.setup()
    u = np.zeros_like(x, dtype=x.dtype)
    # solve
    solver_ = Solver(eqn, x, u, verbose=verbose)
    solver_.maxiter = 10
    solver_.dt = 1e10
    solver_.tol = 1e-6
    
    solver_.param = param_prior + 0j
    solver_.run()
    u_prior = solver_.u.copy()

    model = mc.Model()
    u_benchmark = u_benchmark.copy()
    nsamples = 1000
    with model:
        param =  mc.Normal('param', mu=param_prior, sd=sigma_prior)
        mu = objective(param)
        Y_obs = mc.Normal('Y_obs', mu=mu, sd=sigma_obs, observed=u_benchmark, shape=len(x)) 
        step = mc.Metropolis(vars=[param])
        backend = mc.backends.SQLite("trace.sqlite")
        start = {'param':param_prior}
        trace = mc.sample(nsamples, step=step, start=start, trace=backend)
    mc.summary(trace[int(nsamples/10.0):])
    mc.traceplot(trace[int(nsamples/10.0):])


    # calculate posterior solution
    params = trace["param"][int(nsamples/10.0):]
    param_post = params[-1]
    u = np.zeros_like(x, dtype=x.dtype)
    # solve
    solver_ = Solver(eqn, x, u, verbose=verbose)
    solver_.maxiter = 10
    solver_.dt = 1e10
    solver_.tol = 1e-6
    solver_.param = param_post + 0j
    solver_.run()
    u_post = solver_.u.copy()

    savefig('mcmc.pdf')
    figure()
    plot(x[::5], u_benchmark[::5], 'bo', label="Benchmark")
    plot(x, u_prior, 'g-', label="Prior")
    plot(x, u_post, 'r-', label="Posterior")
    legend(loc=4)
    ylabel('u(x)')
    xlabel('x')
    savefig('u.pdf')

    figure()
    beta_range = np.linspace(0.0, param_prior+1.0, 10000)
    prior_pdf = mlab.normpdf(beta_range, param_prior, sigma_prior)
    kde = gaussian_kde(params)
    # these are the values over wich your kernel will be evaluated
    dist_space = np.linspace(params.min(), params.max(), 10000)
    post_pdf = kde(dist_space)
    subplot(121)
    title('Prior')
    plot(beta_range, prior_pdf/prior_pdf.max(), 'g-', label="Prior")
    subplot(122)
    title('Posterior')
    plot(dist_space, post_pdf/post_pdf.max(), 'r-', label="Posterior")
    savefig('pdfs.pdf')
    show()

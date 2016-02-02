from pylab import *
from solver import Solver
from equation import NonConvexEquationParam
import math
eqn = NonConvexEquationParam()
eqn.setup()
x = np.linspace(0.0, 1.0, 21, endpoint=True, dtype=np.complex)
x = x**1.5
x = pi/2 + x*pi
print x
# intialize solution
u = np.sin(x) + 1.0
solver = Solver(eqn, x, u)
solver.param = np.ones_like(x)*1.0
solver.maxiter = 3
solver.dt = 1e10
ux = solver.calc_first_der(u)
uxx = solver.calc_second_der(u)
uxxx = solver.calc_third_der(u)
uxxxx = solver.calc_fourth_der(u)

figure()
subplot(211)
plot(x[1:-1], solver.u[1:-1], 'gx-')
plot(x[1:-1], np.sin(x[1:-1]), 'r-')

subplot(212)
plot(x[1:-1], abs(solver.u[1:-1] - np.sin(x[1:-1])), 'r-')

figure()
subplot(211)
plot(x[1:-1], ux[1:-1], 'gx-')
plot(x[1:-1], np.cos(x[1:-1]), 'r-')
subplot(212)
plot(x[1:-1], abs(ux[1:-1] - np.cos(x[1:-1])), 'r-')

figure()
subplot(211)
plot(x[1:-1], uxx[1:-1], 'gx-')
plot(x[1:-1], -np.sin(x[1:-1]), 'r-')
subplot(212)
plot(x[1:-1], abs(uxx[1:-1] +np.sin(x[1:-1])), 'r-')

figure()
subplot(211)
plot(x[1:-1], uxxx[1:-1], 'gx-')
plot(x[1:-1], -np.cos(x[1:-1]), 'r-')
subplot(212)
plot(x[1:-1], abs(uxxx[1:-1] +np.cos(x[1:-1])), 'r-')

figure()
subplot(211)
plot(x[1:-1], uxxxx[1:-1], 'gx-')
plot(x[1:-1], np.sin(x[1:-1]), 'r-')
subplot(212)
plot(x[1:-1], abs(uxxxx[1:-1] -np.sin(x[1:-1])), 'r-')

show()

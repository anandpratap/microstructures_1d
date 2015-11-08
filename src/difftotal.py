import sympy as sp
def difftotal(expr, diffby, diffmap):
    # Replace all symbols in the diffmap by a functional form
    fnexpr = expr.subs({s:s(diffby) for s in diffmap})
    # Do the differentiation
    diffexpr = sp.diff(fnexpr, diffby)
    # Replace the Derivatives with the variables in diffmap
    derivmap = {sp.Derivative(v(diffby), diffby):dv 
                for v,dv in diffmap.items()}
    finaldiff = diffexpr.subs(derivmap)
    # Replace the functional forms with their original form
    return finaldiff.subs({s(diffby):s for s in diffmap})

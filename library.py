import numpy as np

def conjugate_gradient(A,b):
    """Out: the solution of the linear equations system Ax=b."""
    x0,epsilon,kmax = np.zeros(b.shape),1e-16,2000
    r0 = A.dot(x0) - b
    w0 = -r0
    for k in range(kmax):
        alphak = -w0.T.dot(r0) / w0.T.dot(A.dot(w0))
        xk = x0 + alphak*w0
        rk = r0 + alphak*A.dot(w0)
        gammak = rk.T.dot(rk) / r0.T.dot(r0)
        wk = -rk + gammak*w0
        if np.sqrt(rk.T.dot(rk)) < epsilon:
            return xk
        else: 
            r0,x0,w0 = rk,xk,wk
    return xk

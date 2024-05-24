import library
import numpy as np
import scipy.linalg as la

#For the example, we construct a system of linear equations.
n = 3
A = np.random.randn(n,n)
A = A.T.dot(A)
sol = np.random.randn(n,1)
b = A.dot(sol)

x0 = library.conjugate_gradient(A, b)
res = la.norm(A.dot(x0)-b)

print('Residual: {0:3.4e}'.format(res))
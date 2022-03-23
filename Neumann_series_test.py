import numpy as np
from numpy.linalg import inv, matrix_power

M = np.matrix('1 0.5; 0.25 1') # matrix to invert
M_inv = inv(M)
print(M_inv)

Id = np.identity(2) 
S_l = np.zeros((2, 2))

l = 20
for i in range(l):
    if i == 0:
        S_l += Id
    else:
        S_l += (Id - M) ** i
    # print(S_l)

print(S_l)

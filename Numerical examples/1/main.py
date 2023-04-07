import numpy as np
import calfem.core as fem

L = 2
element_count = 3
degrees_of_freedom = 4
A = 10
Q = 100
k = 5

e_dof = np.array([[1, 2], [2, 3], [3, 4]])

dof = np.array([1, 2, 3, 4])

K = np.mat(np.zeros((degrees_of_freedom, degrees_of_freedom)))

f = np.mat(np.zeros((degrees_of_freedom, 1)))

for edof in e_dof:
    fem.assem(edof, K, fem.spring1e(k*A/L), f, 2*Q/L)

bc = np.array([1])
bcVal = np.array([0])
q_end = 15

f_b = np.array([[0, 0, 0, -A*q_end]])

a, r = fem.solveq(K, f+f_b.T, bc, bcVal)

print(a)

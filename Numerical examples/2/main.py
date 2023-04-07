import numpy as np
import calfem.core as fem
import scipy.io as sio
import matplotlib.pyplot as plt
from types import SimpleNamespace

k = 1
# Isotropic material
D = k * np.array([[1, 0], [0, 1]])

# From geom2.m exported with >> save('vars.mat', '-v7')
geometry = SimpleNamespace(**sio.loadmat('vars.mat', squeeze_me=True))

K = np.mat(np.zeros((geometry.ndof, geometry.ndof)))
f = np.mat(np.zeros((geometry.ndof, 1)))

for edof, eex, eey in zip(geometry.edof, geometry.ex, geometry.ey):
    te = fem.flw2te(eex, eey, [1], D)
    fem.assem(edof[1:], K, te)

a, _ = fem.solveq(K, f, geometry.bc[:,0], geometry.bc[:,1])

ed = fem.extract_ed(geometry.edof[:,1:], a)

plt.scatter(geometry.ex,geometry.ey,c=ed)
plt.show()

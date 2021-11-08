
import numpy as np
import matplotlib.pyplot as plt

def k_fn(X1, X2):
    norms = np.sqrt((-np.dot(X1,(2*X2.T)) + np.sum(X1*X1,1)[:,None]) + np.sum(X2*X2,1)[None,:])
    return (1+norms)*np.exp(-norms)

X_grid = np.arange(0, 10, 0.5)[:,None]
N_grid = X_grid.shape[0]
K_grid = k_fn(X_grid, X_grid) + 1e-9*np.eye(N_grid)
L_grid = np.linalg.cholesky(K_grid)
plt.figure(1)
plt.clf()
f_grid = np.dot(L_grid, np.random.randn(N_grid))
plt.plot(X_grid, f_grid, '-')
plt.show()



K = np.zeros((N_grid,N_grid))
for i in range(N_grid):
    v = np.column_stack((f_grid[:N_grid-i], f_grid[i:]))
    c = np.cov(v, rowvar=False)
    if i==N_grid-1:
        K[0,i] = np.var(v) #only one row of data
        K[i,0] = np.var(v)
        break
    if i==10:
        pass
    for j in range(N_grid-i):
        K[j,j+i] = c[0,1]
        K[j+i,j] = c[0,1]
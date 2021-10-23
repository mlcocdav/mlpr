import numpy as np
import matplotlib.pyplot as plt

N = 100; D = 1
X = np.random.rand(N, D) - 0.5
yy = X**3

mu = np.random.rand(N)
X = np.tile(mu[:,None], (1, D)) + 0.01*np.random.randn(N, D)
yy = 0.1*np.random.randn(N) + mu
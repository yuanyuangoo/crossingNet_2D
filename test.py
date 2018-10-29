import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import pprint
n=23
mean = np.zeros(n)
cov = np.eye(n)

val_noises = np.random.multivariate_normal(mean, cov, self.batch_size).astype(np.float32)
noises = np.random.multivariate_normal(mean, cov, self.batch_size).astype(np.float32)
pprint.pprint(val_noises.shape)

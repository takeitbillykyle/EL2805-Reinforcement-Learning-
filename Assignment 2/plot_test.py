import matplotlib.pyplot as plt
import numpy as np


sum = 0
for _ in range(10000):
	sum += np.random.geometric(1-29/30)
print(sum/10000)
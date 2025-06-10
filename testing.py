#conda_env: NeuroEvolution
import math
import numpy as np
import matplotlib.pyplot as plt

# Generate right-skewed data (Exponential distribution)
s = np.random.exponential(scale=15, size=100)

# Clip values to stay within [0, 100]
s = s[s <= 100]
for element in s:
    print(math.floor(element))

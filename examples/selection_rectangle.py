import numpy as np
import matplotlib.pyplot as plt

from cloudviz.selection import RectangleSelection

# Generate datapoints
n = 100
x = np.random.random(n)
y = np.random.random(n)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
points = ax.scatter(x, y, c='green')
b = RectangleSelection(ax, points)

import numpy as np
import matplotlib.pyplot as plt

from cloudviz.selection import LassoSelection

# Use a fake subset object to simplify example
class Subset(object):
    def __setattr__(self, attribute, value):
        print "%s has been updated" % attribute
        object.__setattr__(self, attribute, value)

# Generate datapoints
n = 100
x = np.random.random(n)
y = np.random.random(n)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
points = ax.scatter(x, y, c='green')
subset = Subset()
b = LassoSelection(ax, points, subset)

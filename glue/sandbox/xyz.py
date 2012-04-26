import numpy as np
import matplotlib.pyplot as plt

from cloudviz import Client, TabularData, Hub, Data
from cloudviz.data import Component
from cloudviz.subset import ElementSubset


class ScatterWidget(Client):

    def __init__(self, data):

        Client.__init__(self, data)

        self.name = None

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(1,1,1)
        self.xattribute = None
        self.yattribute = None

        self.scatter_object = {}

    def plot_data(self):

        print "Plot data in %s" % self.name

        x = self.data.components[self.xattribute].data.ravel()
        y = self.data.components[self.yattribute].data.ravel()

        if self.data in self.scatter_object:
            self.scatter_object[self.data].remove()
            self.scatter_object.pop(self.data)

        self.scatter_object[self.data] = self.ax.scatter(x, y, color='k', zorder=10, s=10)

    def plot_subsets(self):

        print "Plot subsets in %s" % self.name

        for subset in self.data.subsets:
            self.plot_subset(subset)

        self.refresh()

    def plot_subset(self, subset):

        print "Plot subset in %s" % self.name

        print "Plotting %i points" % np.sum(subset.mask)

        x = self.data.components[self.xattribute].data.ravel()
        y = self.data.components[self.yattribute].data.ravel()

        self.remove_subset(subset)

        if subset.mask.sum() > 0: 
            self.scatter_object[subset] = self.ax.scatter(x[subset.mask], y[subset.mask], color='r', zorder=20, s=5)

        self.refresh()

    def remove_subset(self, subset):

        if subset in self.scatter_object:
            self.scatter_object[subset].remove()
            self.scatter_object.pop(subset)

        self.refresh()

    def refresh(self):
        print "refresh in %s" % self.name
        self.figure.canvas.draw()
        
    def _add_subset(self, subset):
        print "_add_subset in %s" % self.name
        self.plot_subset(subset)

    def _remove_subset(self, subset):
        print "_remove_subset in %s" % self.name
        self.remove_subset(subset)
        
    def _update_all(self, attribute=None):
        print "_update_all in %s" % self.name
        self.plot_data()
        self.plot_subsets()

    def _update_subset(self, subset, attribute=None):
        print "_update_subset in %s with attribute=%s" % (self.name, attribute)
        self.plot_subset(subset)

shape = (1000,)
data = Data()
data.components['x'] = Component(np.random.random(shape))
data.components['y'] = Component(np.random.random(shape))
data.components['z'] = Component(np.random.random(shape))
data.shape = shape
data.ndim = 1

w1 = ScatterWidget(data)
w1.xattribute = 'x'
w1.yattribute = 'y'
w1.name = "Widget 1"

w2 = ScatterWidget(data)
w2.xattribute = 'y'
w2.yattribute = 'z'
w2.name = "Widget 2"

h = Hub()
h.add_client(w1)
h.add_client(w2)

subset = ElementSubset(data)
subset.mask = np.random.random(shape) > 0.9
print np.sum(subset.mask)

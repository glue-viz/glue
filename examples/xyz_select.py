import numpy as np
import matplotlib.pyplot as plt

from cloudviz import Client, TabularData, Hub, Data
from cloudviz.data import Component
from cloudviz.subset import ElementSubset
from cloudviz.selection import LassoSelection


class ScatterWidget(Client):

    def __init__(self, data):

        Client.__init__(self, data)

        self.name = None

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.xattribute = None
        self.yattribute = None

        self.scatter_object = {}

        self.figure.canvas.mpl_connect('button_press_event', self.new_selection)
        self.selection = None

    def plot_data(self):

        print "Plot data in %s" % self.name

        x = self.data.components[self.xattribute].data.ravel()
        y = self.data.components[self.yattribute].data.ravel()

        if self.data in self.scatter_object:
            self.scatter_object[self.data].remove()
            self.scatter_object.pop(self.data)

        if self.data in self.scatter_object:
            self.scatter_object[self.data].set_color('k')
        else:
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
            if subset in self.scatter_object:
                self.scatter_object[subset].set_color('r')
            else:
                self.scatter_object[subset] = self.ax.scatter(x[subset.mask], y[subset.mask], color='r', zorder=20, s=5)

        self.refresh()

    def remove_subset(self, subset):

        if subset in self.scatter_object:
            self.scatter_object[subset].remove()
            self.scatter_object.pop(subset)

        self.refresh()

    def new_selection(self, event):
        if event.button != 3: 
            return
        if self.selection is not None:
            return
        if len(self.data.subsets) == 0:
            subset = ElementSubset(self.data)
        else:
            subset = self.data.subsets[0]
        subset.register()
        self.selection = LassoSelection(self.ax, self.scatter_object[self.data], subset)

    def refresh(self):
        print "refresh in %s" % self.name
        self.figure.canvas.draw()

    def _add_subset(self, message):
        print "_add_subset in %s" % self.name
        subset = message.subset
        self.plot_subset(subset)

    def _remove_subset(self, message):
        print "_remove_subset in %s" % self.name
        subset = message.subset
        self.remove_subset(subset)

    def _update_all(self, message):
        print "_update_all in %s" % self.name
        self.plot_data()
        self.plot_subsets()

    def _update_subset(self, message):
        print "_update_subset in %s with attribute=%s" % (self.name, message.attribute)
        subset = message.subset
        self.plot_data()
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
w1.register_to_hub(h)
w2.register_to_hub(h)
data.register_to_hub(h)

w1.plot_data()
w2.plot_data()

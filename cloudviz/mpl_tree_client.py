from cloudviz.tree_client import TreeClient

import matplotlib.pyplot as plt

class MplTreeClient(TreeClient):
    """ A tree client based on matplotlib """

    def __init__(self, data, layout=None, figure=None, axes=None):
        super(MplTreeClient, self).__init__(data, layout=layout)

        #set up default axes and figure objects
        if axes is not None and figure is not None and \
                axes.figure is not figure:
            raise Exception("Axes does not belong to figure")

        if axes is not None:
            self._axes = axes
            self._figure = axes.figure
        else:
            self._figure = figure
            if self._figure is None:
                self._figure = plt.figure()
            self._axes = self._figure.add_subplot(1, 1, 1)

    def _remove_subset(self, message):
        """ Erase a subset from the plot """
        s = message.subset
        if s in self._plots:
            self._plots[s].remove()

        super(MplTreeClient, self)._remove_subset(message)

    def _update_data_plot(self):
        """ update state information for main data plot """
        data = self.layout.tree_to_xy(self.data.tree)

        # updating for the first time
        if self.data not in self._plots:
            p = self._axes.plot(data[0], data[1], linewidth=2, picker=5)[0]
            self._plots[self.data] = p

        p = self._plots[self.data]
        p.set_data(data)

    def _update_subset_single(self, subset):
        """ update state information for a subset data plot """

        if subset not in self.data.subsets:
            raise TypeError("Input is not one of data's subsets: %s" % subset)

        #empty subset
        if not subset.node_list:
            if subset in self._plots:
                self._plots[subset].set_visible(False)
            return

        # convert tree into a line plot
        x, y = self.layout.branch_to_xy(subset.node_list)

        if subset not in self._plots:
            plot = self._axes.plot(x, y, linewidth=4)[0]
            self._plots[subset] = plot
        else:
            self._plots[subset].set_data((x, y))

        # update plot visual properties
        self._plots[subset].set_visible(True)
        self._plots[subset].set_color(subset.style.color)

    def _update_axis_labels(self):
        pass

    def _redraw(self):
        self._figure.canvas.draw()

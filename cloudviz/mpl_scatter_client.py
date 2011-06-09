from cloudviz.scatter_client import ScatterClient
import matplotlib.pyplot as plt


class MplScatterClient(ScatterClient):

    def __init__(self, data, figure=None, axes=None):
        if axes is not None and figure is not None and \
                axes.figure is not figure:
            raise Exception("Axes and figure are incompatible")

        ScatterClient.__init__(self, data)

        if axes is not None:
            self._ax = axes
            self._figure = axes.figure
        else:
            if figure is None:
                self._figure = plt.figure()
            self._ax = self._figure.add_subplot(1, 1, 1)

    def _redraw(self):
        """
        Re-render the screen
        """
        self._figure.canvas.draw()

    def _remove_subset(self, message):

        s = message.subset
        if s in self._plots:
            self._plots[s].remove()

        super(MplScatterClient, self)._remove_subset(message)

    def _update_axis_labels(self):
        """
        Sync the axis labels to reflect which components are currently
        being plotted
        """

        xlabel = self._xatt
        if xlabel is not None and \
                self.data.components[self._xatt].units is not None:
            xlabel += ' (%s)' % self.data.components[self._xatt].units

        ylabel = self._yatt
        if ylabel is not None and \
                self.data.components[self._yatt].units is not None:
            ylabel += ' (%s)' % self.data.components[self._yatt].units

        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

    def _update_data_plot(self):
        """
        Sync the location of the scatter points to
        reflect what components are being plotted
        """
        if self._xdata is None or self._ydata is None:
            return
        if self.data not in self._plots:
            plot = self._ax.scatter(self._xdata, self._ydata, color='k')
            self._plots[self.data] = plot
        else:
            self._plots[self.data].set_offsets(
                zip(self._xdata, self._ydata))

    def _update_subset_single(self, s):
        """
        Update the location and visual properties
        of each point in a single subset

        Parameters:
        ----------
        s: A subset instance
        The subset to refresh.

        """
        if self._xdata is None or self._ydata is None:
            return

        if s not in self.data.subsets:
            raise Exception("Input is not one of data's subsets: %s" % s)

        # handle special case of empty subset
        if s.to_mask().sum() == 0:
            if s in self._plots:
                self._plots[s].set_visible(False)
            return

        if s not in self._plots:
            plot = self._ax.scatter(self._xdata[s.to_mask()],
                              self._ydata[s.to_mask()], s=5)
            self._plots[s] = plot
        else:
            self._plots[s].set_offsets(
                zip(self._xdata[s.to_mask()], self._ydata[s.to_mask()]))

        self._plots[s].set_visible(True)
        self._plots[s].set(**s.style)


if __name__ == "__main__":
    """
    This is a self contained (non-interactive) example of setting up
    a cloudviz environment with 2 catalog clients linked to the same data.
    """
    import cloudviz as cv
    from time import sleep
    import sys

    #set up the data
    d = cv.TabularData()
    d.read_data(sys.path[0] + '/../examples/oph_c2d_yso_catalog.tbl')

    # create the hub
    h = cv.Hub()

    # create the 2 clients
    c = MplScatterClient(d)
    c2 = MplScatterClient(d)

    # register the clients and data to the hub
    # (to receive and send events, respectively)
    c.register_to_hub(h)
    c2.register_to_hub(h)
    d.register_to_hub(h)

    #define the axes to plot
    c.set_xdata('ra')
    c.set_ydata('dec')
    c2.set_xdata('IR1_flux_1')
    c2.set_ydata('IR2_flux_1')
    sleep(1)

    # create a new subset. Note we need to register() each subset
    mask = d.components['ra'].data > 248
    s = cv.subset.ElementSubset(d, mask=mask)
    s.register()
    sleep(1)

    #change plot properties. Updated autmatically
    s.style['color'] = 'm'
    s.style['alpha'] = .8
    sleep(1)

    #change one of the axes. Automatically updated
    c2.set_ydata('IR3_flux_1')
    sleep(1)

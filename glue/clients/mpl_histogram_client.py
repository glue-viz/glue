from glue.histogram_client import HistogramClient
from glue.viz_client import init_mpl
import matplotlib.pyplot as plt


class MplHistogramClient(HistogramClient):

    def __init__(self, data, figure=None, axes=None, options=None):

        HistogramClient.__init__(self, data, options=options)
        self._figure, self._ax = init_mpl(figure, axes)

        self.render_all = False #  draw the histgram of the whole data set?
        self.all_color = 'black' #  color for the background histogram
        self.options['histtype'] = 'barstacked' #  histogram style for subset plots

    def _redraw(self):
        """
        Re-render the screen
        """

        # get data and colors
        keys = [k for k in self._plots.keys() if k is not self.data]
        subset_data = [self._plots[k] for k in keys]
        colors = [k.style.color
                  if k in self.data.subsets else 'black'
                  for k in keys]

        for p in self._ax.patches:
            p.remove()

        # draw histogram for the whole data set
        if self.render_all and self.data in self._plots:
            base = self._ax.hist(self._plots[self.data].ravel(),
                                 color = self.all_color,
                                 **self.options)
            bins = base[1]

        # overplot the subsets
        if subset_data:
            if bins is not None:
                self._ax.hist(subset_data, bins=bins, color=colors, **self.options)
            else:
                self._ax.hist(subset_data, color=colors, **self.options)

        # render
        self._figure.canvas.draw()

    def _remove_subset(self, message):

        s = message.subset
        if s in self._plots:
            self._plots.remove(s)

        super(MplHistogramClient, self)._remove_subset(self, message)

    def _update_data_plot(self):
        """
        Sync the main histogram
        """

        if self._component is None:
            return

        x = self.data[self._component].data
        self._plots[self.data] = x

    def _update_axis_labels(self):
        xtitle = self._component
        if self.data[self._component].units:
            xtitle += ' ( %s )' % self.data[self._component].units

        self._ax.set_xlabel(xtitle)
        self._ax.set_ylabel('N')

    def _update_subset_single(self, s):
        """
        Update the location and visual properties
        of each point in a single subset

        Parameters:
        ----------
        s: A subset instance
        The subset to refresh.

        """

        if self._component is None:
            return

        if s not in self.data.subsets:
            raise KeyError("Input is not one of data's subsets: %s" % s)

        mask = s.to_mask(data = self.data)
        if mask.sum() == 0:
            self._plots[s] = []
            return

        x = self.data[self._component].data
        x = x[s.to_mask(data = self.data)]
        self._plots[s] = x

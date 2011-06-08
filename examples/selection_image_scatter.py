"""
This is a self contained (non-interactive) example of setting up
a cloudviz environment with 2 catalog clients linked to the same data.
"""
import cloudviz as cv

#set up the data
d = cv.GriddedData()
d.read_data('test.fits')
d.components['INDEX_MAP'] = cv.Component((d.components['PRIMARY'].data > 3).astype(float))

# create the hub
h = cv.Hub()

# create the 2 clients
c = cv.MplScatterClient(d)
c2 = cv.MplImageClient(d)

# register the clients and data to the hub
# (to receive and send events, respectively)
c.register_to_hub(h)
c2.register_to_hub(h)
d.register_to_hub(h)

# #define the axes to plot
c.set_xdata('PRIMARY')
c.set_ydata('INDEX_MAP')
c2.set_component('PRIMARY')

# create a new subset. Note we need to register() each subset
s = cv.subset.ElementSubset(d)
s.mask = d.components['INDEX_MAP'].data > 0.5
s.register()

#change plot properties. Updated autmatically
s.style['color'] = 'm'
s.style['alpha'] = .8


class Selection(object):

    def __init__(self, ax, client, subset):
        self.ax = ax
        self.subset = subset
        self.client = client
        self.roi = cv.roi.MplRoi(ax)
        self.client._figure.canvas.mpl_connect('button_press_event',
                                                self.button_press_event)

        self.client._figure.canvas.mpl_connect('motion_notify_event',
                                                self.motion_notify_event)

        self.client._figure.canvas.mpl_connect('button_release_event',
                                                self.button_release_event)

    def button_press_event(self, event, **kwargs):
        """ Button presses start a new roi definition"""
        if not event.inaxes:
            return
        self.roi.reset()
        self.roi.add_point(event.xdata, event.ydata)

    def motion_notify_event(self, event, **kwargs):
        """ Button motion adds new points to the roi"""
        if not event.inaxes or event.button is None:
            return
        self.roi.add_point(event.xdata, event.ydata)

    def button_release_event(self, event, **kwargs):
        """ Button releases translate the roi to a subset"""
        mask = self.roi.contains(self.client._xdata, self.client._ydata)
        self.subset.mask = mask.reshape(s.mask.shape)

s1 = Selection(c._ax, c, s)
s2 = Selection(c2._ax, c2, s)

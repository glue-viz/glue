import cloudviz as cv
from catalog_client import CatalogClient


""" An example of using the catalog client with a lasso-based ROI
object.  For simplicity, all mouse events are sent to global functions
which update the ROI

In a real application, the client would manage the ROI object manually
"""


def button_press_event(event, **kwargs):
    """ Button presses start a new roi definition"""
    if not event.inaxes:
        return
    roi.reset()
    roi.add_point(event.xdata, event.ydata)


def motion_notify_event(event, **kwargs):
    """ Button motion adds new points to the roi"""
    if not event.inaxes:
        return
    roi.add_point(event.xdata, event.ydata)


def button_release_event(event, **kwargs):
    """ Button releases translate the roi to a subset"""
    mask = roi.contains(c2._xdata, c2._ydata)
    s.mask = mask


# set up the data
d = cv.TabularData()
d.read_data('oph_c2d_yso_catalog.tbl')

# create the hub
h = cv.Hub()

# create the 2 clients
c = CatalogClient(d)
c2 = CatalogClient(d)
c.set_xdata('ra')
c.set_ydata('dec')
c2.set_xdata('IR1_flux_1')
c2.set_ydata('IR2_flux_1')

# create a subset
s = cv.subset.ElementSubset(d)
mask = d.components['ra'].data.ravel() > 248
s.mask = mask

# register clients and data to hub
# (to receive and send events, respectively)
# register subset to send events
c.register_to_hub(h)
c2.register_to_hub(h)
d.register_to_hub(h)
s.register()

# create an ROI. Attacht to c2
roi = cv.roi.MplRoi(c2._ax)
c2._figure.canvas.mpl_connect('button_press_event',
                              button_press_event)

c2._figure.canvas.mpl_connect('motion_notify_event',
                              motion_notify_event)

c2._figure.canvas.mpl_connect('button_release_event',
                              button_release_event)

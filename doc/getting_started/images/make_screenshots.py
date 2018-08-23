# This is a script that can be used to reproduce the screenshots for the
# Getting Started guide. The idea is that as we update glue, we can easily
# regenerate screenshots to make sure we include the latest ui.

from glue.app.qt import GlueApplication
from glue.viewers.image.qt import ImageViewer
from glue.viewers.histogram.qt import HistogramViewer
from glue.viewers.scatter.qt import ScatterViewer
from glue.core.edit_subset_mode import AndNotMode, ReplaceMode
from glue.core.link_helpers import LinkSame

ga = GlueApplication()
ga.resize(1230, 900)
ga.show()

ga.app.processEvents()
ga.screenshot('main_window1.png')

image = ga.load_data('w5.fits')
image.label = 'W5'

ga.app.processEvents()
ga.screenshot('data_open.png')

image_viewer = ga.new_data_viewer(ImageViewer, data=image)
image_viewer._mdi_wrapper.resize(450, 400)

image_viewer.state.layers[0].v_min = 440
image_viewer.state.layers[0].v_max = 900
image_viewer.state.layers[0].stretch = 'sqrt'
image_viewer.state.reset_limits()

ga.app.processEvents()
ga.screenshot('main_window2.png')

py, px = image.pixel_component_ids
subset_state = (px > 500) & (px < 900) & (py > 300) & (py < 800)

ga.data_collection.new_subset_group(subset_state=subset_state, label='Subset 1')

ga.app.processEvents()
ga.screenshot('w5_west.png')

histogram_viewer = ga.new_data_viewer(HistogramViewer, data=image)
histogram_viewer._mdi_wrapper.resize(450, 400)
histogram_viewer._mdi_wrapper.move(450, 0)
histogram_viewer.state.x_min = 400
histogram_viewer.state.x_max = 700
histogram_viewer.state.update_bins_to_view()
histogram_viewer.state.normalize = True

ga.app.processEvents()

ga.session.edit_subset_mode.mode = AndNotMode
cid = image.main_components[0]
subset_state = (cid >= 450) & (cid <= 500)
ga.session.edit_subset_mode.update(ga.data_collection, subset_state)

ga.app.processEvents()
ga.screenshot('subset_refine.png')

catalog = ga.load_data('w5_psc.vot')
catalog.label = 'Point Sources'

# Set up links
link1 = LinkSame(image.id['Right Ascension'], catalog.id['RAJ2000'])
link2 = LinkSame(image.id['Declination'], catalog.id['DEJ2000'])
ga.data_collection.add_link(link1)
ga.data_collection.add_link(link2)

scatter_viewer = ga.new_data_viewer(ScatterViewer, data=catalog)
scatter_viewer._mdi_wrapper.resize(900, 400)
scatter_viewer._mdi_wrapper.move(0, 400)
scatter_viewer.state.x_att = catalog.id['__4.5__-__5.8_']
scatter_viewer.state.y_att = catalog.id['__4.5_']
scatter_viewer.state.x_min = -1
scatter_viewer.state.x_max = 1.6
scatter_viewer.state.y_min = 1
scatter_viewer.state.y_max = 17

ga.session.edit_subset_mode.mode = ReplaceMode
subset_state = (px > 300) & (px < 400) & (py > 600) & (py < 700)
ga.session.edit_subset_mode.update(ga.data_collection, subset_state)

ga.app.processEvents()
ga.screenshot('link_subset_1.png')

image_viewer.add_subset(catalog.subsets[0])

# FIXME: the following doesn't work currently
image_viewer.axes.set_xlim(250, 450)
image_viewer.axes.set_ylim(550, 750)

ga.app.processEvents()
ga.screenshot('link_subset_2.png')

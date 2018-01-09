from glue.core.data_factories import load_data
from glue.core import DataCollection
from glue.core.link_helpers import LinkSame
from glue.app.qt.application import GlueApplication

#load 2 datasets from files
image = load_data('w5.fits')
catalog = load_data('w5_psc.vot')
dc = DataCollection([image, catalog])

# link positional information
dc.add_link(LinkSame(image.id['Right Ascension'], catalog.id['RAJ2000']))
dc.add_link(LinkSame(image.id['Declination'], catalog.id['DEJ2000']))

#start Glue
app = GlueApplication(dc)
app.start()

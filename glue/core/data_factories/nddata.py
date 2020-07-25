from astropy.nddata import NDDataBase, NDData

from glue.config import data_factory
from glue.core.data import Data
from glue.core.component import Component
from glue.core.data_collection import DataCollection
from glue.core.data_factories.fits import is_fits, fits_reader

__all__ = ['is_nddata', 'nddata_data_loader', 'formatted_nddata_factory']


def is_nddata(data):
    """
    To check whether the data is of the NDData type.
    """
    if isinstance(data, NDDataBase):
        return True
    else:
        return False


@data_factory(label="NDData", identifier=is_nddata)
def nddata_data_loader(nddata, *args, **kwargs):
    """
    Build a data set from nddata.
    """
    result = Data()
    ndd = Component(nddata.data, units=nddata.unit)
    result.meta = nddata.meta
    result.mask = nddata.mask
    result.uncertainty = nddata.uncertainty
    result.add_component(ndd, label="data")

    return result


def formatted_nddata_factory(format, label):

    @data_factory(label=label, identifier=is_fits)
    def factory(filepath, **kwargs):
        extensions = fits_reader(filepath)
        dc = DataCollection()
        for hdu in extensions:
            ndd = NDData(hdu.data,
                         mask=hdu.mask,
                         unit=hdu.unit,
                         uncertainty=hdu.uncertainty,
                         meta=hdu.meta)
            result = nddata_data_loader(ndd)
            dc.append(result)
        return dc

    factory.__name__ = '%s_factory' % format

    return factory


fits_to_nddata_factory = formatted_nddata_factory('fits', 'Fits as NDData')

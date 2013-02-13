from ..external.qt.QtCore import QMimeData


class PyMimeData(QMimeData):
    """Stores references to live python objects.

    Normal QMimeData instances store all data as QByteArrays. This
    makes it hard to pass around live python objects in drag/drop
    events, since one would have to convert between object references
    and byte sequences.

    The object to store is passed to the constructor, and stored in
    the application/py_instance mime_type.

    Additional custom python objects can be stored by passing extra
    keyword arguments to __init__
    """
    MIME_TYPE = 'application/py_instance'

    def __init__(self, instance, **kwargs):
        """
        :param instance: The python object to store

        kwargs: Optional mime type / objects pairs to store as objects
        """
        super(PyMimeData, self).__init__()

        self._instances = {self.MIME_TYPE: instance}
        self.setData(self.MIME_TYPE, '1')

        for k, v in kwargs.items():
            print 'setting %s to %s' % (k, v)
            self.setData(k, '1')
            assert self.hasFormat(k)
            self._instances[k] = v

    def data(self, mime_type):
        """ Retrieve the data stored at the specified mime_type

        If mime_type is application/py_instance, a python object
        is returned. Otherwise, a QByteArray is returned """
        if str(mime_type) in self._instances:
            return self._instances[mime_type]

        return super(PyMimeData, self).data(mime_type)

LAYER_MIME_TYPE = 'glue/layer'
LAYERS_MIME_TYPE = 'glue/layers'

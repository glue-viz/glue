from __future__ import absolute_import, division, print_function

from ..external.qt.QtCore import QMimeData, QByteArray


class PyMimeData(QMimeData):
    """
    A custom MimeData object that stores live python objects

    Associate specific objects with a mime type by passing
    mime type / object kev/value pairs to the __init__ method

    If a single object is passed to the init method, that
    object is associated with the PyMimeData.MIME_TYPE mime type
    """
    MIME_TYPE = 'application/py_instance'

    def __init__(self, instance=None, **kwargs):
        """
        :param instance: The python object to store

        kwargs: Optional mime type / objects pairs to store as objects
        """
        super(PyMimeData, self).__init__()

        self._instances = {}

        self.setData(self.MIME_TYPE, instance)
        for k, v in kwargs.items():
            self.setData(k, v)

    def formats(self):
        return list(set(super(PyMimeData, self).formats() +
                        list(self._instances.keys())))

    def hasFormat(self, fmt):
        return fmt in self._instances or super(PyMimeData, self).hasFormat(fmt)

    def setData(self, mime, data):
        super(PyMimeData, self).setData(mime, QByteArray('1'))
        self._instances[mime] = data

    def data(self, mime_type):
        """ Retrieve the data stored at the specified mime_type

        If mime_type is application/py_instance, a python object
        is returned. Otherwise, a QByteArray is returned """
        if str(mime_type) in self._instances:
            return self._instances[mime_type]

        return super(PyMimeData, self).data(mime_type)


#some standard glue mime types
LAYER_MIME_TYPE = 'glue/layer'
LAYERS_MIME_TYPE = 'glue/layers'
INSTANCE_MIME_TYPE = PyMimeData.MIME_TYPE

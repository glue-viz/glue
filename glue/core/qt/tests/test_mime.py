from ..mime import GlueMimeListWidget, LAYERS_MIME_TYPE


class TestGlueMimeListWidget(object):

    def setup_method(self, method):
        self.w = GlueMimeListWidget()

    def test_mime_type(self):
        assert self.w.mimeTypes() == [LAYERS_MIME_TYPE]

    def test_mime_data(self):
        self.w.set_data(3, 'test data')
        self.w.set_data(4, 'do not pick')
        mime = self.w.mimeData([3])
        mime.data(LAYERS_MIME_TYPE) == ['test data']

    def test_mime_data_multiselect(self):
        self.w.set_data(3, 'test data')
        self.w.set_data(4, 'also pick')
        mime = self.w.mimeData([3, 4])
        mime.data(LAYERS_MIME_TYPE) == ['test data', 'also pick']

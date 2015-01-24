from ..misc import as_variable_name, file_format


def test_as_variable_name():
    def check(input, expected):
        assert as_variable_name(input) == expected

    tests = [('x', 'x'),
             ('x2', 'x2'),
             ('2x', '_2x'),
             ('x!', 'x_'),
             ('x y z', 'x_y_z'),
             ('_XY', '_XY')
             ]
    for input, expected in tests:
        yield check, input, expected


class TestFileFormat(object):

    def test_gz(self):
        fmt = file_format('test.tar.gz')
        assert fmt == 'tar'

    def test_normal(self):
        fmt = file_format('test.data')
        assert fmt == 'data'

    def test_underscores(self):
        fmt = file_format('test_file.fits_file')
        assert fmt == 'fits_file'

    def test_multidot(self):
        fmt = file_format('test.a.b.c')
        assert fmt == 'c'

    def test_nodot(self):
        fmt = file_format('test')
        assert fmt == ''

import unittest

import numpy as np

import glue
from glue.util import *

class TestRelim(unittest.TestCase):
    pass

class TestFileFormat(unittest.TestCase):

    def test_gz(self):
        fmt = file_format('test.tar.gz')
        self.assertEquals(fmt, 'tar')

    def test_normal(self):
        fmt = file_format('test.data')
        self.assertEquals(fmt, 'data')

    def test_underscores(self):
        fmt = file_format('test_file.fits_file')
        self.assertEquals(fmt, 'fits_file')

    def test_multidot(self):
        fmt = file_format('test.a.b.c')
        self.assertEquals(fmt, 'c')


if __name__ == "__main__":
    unittest.main()
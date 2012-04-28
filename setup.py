#!/usr/bin/env python

from distutils.core import setup

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

    setup(name='Glue',
          version='0.1.0',
          packages=['glue', 'glue.qt'],
          cmdclass={'build_py': build_py},
          package_data={'glue': ['examples/*']}
      )

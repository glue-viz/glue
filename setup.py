#!/usr/bin/env python

from __future__ import print_function
from distutils.core import setup, Command
from glob import glob
import os

cmdclass = {}

scripts = glob(os.path.join('scripts', '*'))

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys, subprocess
        path = os.path.join('scripts', 'runtests.py')
        errno = subprocess.call([sys.executable, path, 'glue'])
        raise SystemExit(errno)

cmdclass['test'] = PyTest

class BuildQt(Command):

    user_options = [
    ('pyrcc4=', 'p', "Custom pyrcc4 command")
    ]

    def initialize_options(self):
        self.pyrcc4 = 'pyrcc4'

    def finalize_options(self):
        pass

    def run(self):

        import os
        import glob
        from PyQt4.uic import compileUi

        for infile in glob.glob(os.path.join('glue', 'qt', 'ui', '*.ui')):
            print("Compiling " + infile)
            directory, filename = os.path.split(infile)
            outfile = os.path.join(directory, filename.replace('.ui', '.py'))
            compileUi(infile, open(outfile, 'wb'))

        import sys
        import subprocess
        from shutil import copyfile

        print("Compiling glue/qt/glue.qrc")
        if sys.version_info[0] == 2:
            option = '-py2'
        else:
            option = '-py3'
        try:
            subprocess.call([self.pyrcc4, option, 'glue/qt/glue.qrc', '-o', 'glue/qt/glue_qt_resources.py'])
        except OSError:
            print("pyrcc4 command failed - make sure that pyrcc4 "
                  "is in your $PATH, or specify a custom command with "
                  "--pyrcc4=command")

        #XXX Hack: pyuic seems to expect glue/qt/ui/glue_rc.py when
        #loading icons. Copy it there
        copyfile('glue/qt/glue_qt_resources.py', 'glue/qt/ui/glue_rc.py')


cmdclass['build_qt'] = BuildQt

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

class build(build_py):
    def run(self):
        self.run_command("build_qt")
        build_py.run(self)

cmdclass['build_py'] = build

setup(name='Glue',
      version='0.1.0',
      packages=['glue', 'glue.qt', 'glue.core', 'glue.qt.widgets',
               'glue.qt.ui', 'glue.clients', 'glue.tests', 'glue.core.tests',
               'glue.clients.tests', 'glue.qt.tests',
               'glue.qt.widgets.tests'],
      cmdclass=cmdclass,
      package_data={'glue': ['examples/*']},
      scripts=scripts
  )

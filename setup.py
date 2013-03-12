#!/usr/bin/env python
from __future__ import print_function
from distutils.core import setup, Command
from distutils.command.install_scripts import install_scripts
from glob import glob
import os
import sys
import platform
import subprocess

from setupext import (print_line, print_raw, print_status,
                      check_for_numpy, check_for_matplotlib,
                      check_for_qt4, check_for_ipython, check_for_scipy,
                      check_for_astropy, check_for_aplpy, check_for_pytest,
                      check_for_mock, check_for_pil, check_for_atpy,
                      check_for_pyside,
                      )

def is_windows():
    return platform.system() == 'Windows'

def print_sysinfo():
    """Print information about relevant dependencies"""
    #get version information
    for line in open('glue/version.py'):
        if (line.startswith('__version__')):
            exec(line.strip())

    #Print external package information
    print_line()
    print_raw("BUILDING GLUE")
    print_status('glue', __version__)
    print_status('python', sys.version)
    print_status('platform', sys.platform)
    if sys.platform == 'win32':
        print_status('Windows version', sys.getwindowsversion())

    print_raw("")
    print_raw("REQUIRED DEPENDENCIES")
    if not check_for_numpy('1.4'):
        sys.exit(1)


    print_raw("")
    print_raw("RECOMMENDED DEPENDENCIES")
    check_for_matplotlib()
    check_for_qt4()
    check_for_pyside()
    check_for_scipy()

    print_raw("")
    print_raw("OPTIONAL DEPENDENCIES : GENERAL")
    check_for_ipython()
    check_for_pil()

    print_raw("")
    print_raw("OPTIONAL DEPENDENCIES : ASTRONOMY")
    check_for_astropy()
    check_for_atpy()
    check_for_aplpy()

    print_raw("")
    print_raw("OPTIONAL DEPENDENCIES : TESTING")
    check_for_pytest()
    check_for_mock()
    print_line()

cmdclass = {}

scripts = glob(os.path.join('scripts', '*'))


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        path = os.path.join('scripts', 'runtests.py')
        errno = subprocess.call([sys.executable, path, 'glue'])
        raise SystemExit(errno)

cmdclass['test'] = PyTest

def has_qt4():
    """Check if PyQt4 is installed, but do not import"""
    import imp
    try:
        imp.find_module('PyQt4')
        return True
    except ImportError:
        return False

class BuildQt(Command):

    user_options = [
        ('rcc=', 'r', "Custom rcc command (usually pyside-rcc or pyrcc4)"),
        ('uic=', 'u', 'Custom uic command (usually pyside-uic or pyuic4)')
    ]

    def initialize_options(self):
        """Select between PyQt4 and PySide tools"""
        use_pyside = os.environ.get('QT_API', None) == 'pyside'
        use_pyside = use_pyside or not has_qt4()
        if use_pyside:
            print("Using PySide tools to build Qt interfaces")
            self.pyrcc4 = 'pyside-rcc'
            self.pyuic = 'pyside-uic'
        else:
            print("Using PyQt4 tools to build Qt interfaces")
            self.pyrcc4 = 'pyrcc4'
            self.pyuic = 'pyuic4'

    def finalize_options(self):
        pass

    def _make_qt_agnostic(self, fname):
        """Remove Qt4/PySide specific imports"""
        with open(fname) as infile:
            val = infile.read()
        val = val.replace('PyQt4', 'glue.external.qt')
        val = val.replace('PySide', 'glue.external.qt')
        with open(fname, 'wb') as out:
            out.write(val)

    def _compile_ui(self, infile, outfile):
        from cStringIO import StringIO

        try:
            subprocess.call([self.pyuic, infile, '-o', outfile])
        except OSError:
            #note: pyuic4 may be named like pyuic4-2.7
            if self.pyuic == 'pyuic4':
                self.pyuic = 'pyuic4-%i.%i' % sys.version_info[0:2]
                print('falling back to %s' % self.pyuic)
                self._compile_ui(infile, outfile)
                return

            print("uic command failed - make sure that pyuic4 or pyside-uic "
                  "is in your $PATH, or specify a custom command with "
                  "--uic=command")

        self._make_qt_agnostic(outfile)

    def _build_rcc(self, infile, outfile):
        if sys.version_info[0] == 2:
            option = '-py2'
        else:
            option = '-py3'
        try:
            subprocess.call([self.pyrcc4, option, infile, '-o',
                             outfile])
        except OSError:
            #note: pyrcc4 may be named like  pyrcc4-2.7
            if self.pyrcc4 == 'pyrcc4':
                self.pyrcc4 = 'pyrcc4-%i.%i' % sys.version_info[0:2]
                print('falling back to %s' % self.pyrcc4)
                self._build_rcc(infile, outfile)
                return

            print("rcc command failed - make sure that pyrcc4 "
                  "or pyside-rcc4 is in your $PATH, or specify "
                  "a custom command with --rcc=command")

        self._make_qt_agnostic(outfile)

    def run(self):
        from shutil import copyfile

        #compile ui files
        for infile in glob(os.path.join('glue', 'qt', 'ui', '*.ui')):
            print("Compiling " + infile)
            directory, filename = os.path.split(infile)
            outfile = os.path.join(directory, filename.replace('.ui', '.py'))
            self._compile_ui(infile, outfile)

        #build qt resource files
        print("Compiling glue/qt/glue.qrc")
        infile = os.path.join('glue', 'qt', 'glue.qrc')
        outfile = os.path.join('glue', 'qt', 'glue_qt_resources.py')
        self._build_rcc(infile, outfile)

        #Hack: pyuic seems to expect glue/qt/ui/glue_rc.py when
        #loading icons. Copy it there
        copyfile(outfile, os.path.join('glue', 'qt', 'ui', 'glue_rc.py'))


cmdclass['build_qt'] = BuildQt

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py


class build(build_py):
    def run(self):
        print_sysinfo()
        self.run_command("build_qt")
        build_py.run(self)

class glue_install_scripts(install_scripts):
    #on windows, make a glue.bat file
    #lets users just type "glue", instead of "python glue"
    def run(self):
        install_scripts.run(self)
        if not is_windows():
            return
        for script in self.get_outputs():
            if not script.endswith('glue'):
                continue
            bat = "@echo off\n python %s %%*" % script
            outfile = script + '.bat'
            with open(outfile, 'w') as out:
                out.write(bat)

cmdclass['build_py'] = build
cmdclass['install_scripts'] = glue_install_scripts

setup(name='Glue',
      version='0.1.0',
      packages=['glue', 'glue.external', 'glue.qt', 'glue.core', 'glue.qt.widgets',
                'glue.qt.ui', 'glue.clients', 'glue.tests', 'glue.core.tests',
                'glue.clients.tests', 'glue.qt.tests',
                'glue.qt.widgets.tests'],
      cmdclass=cmdclass,
      package_data={'glue': ['examples/*', 'logo.png']},
      scripts=scripts
      )

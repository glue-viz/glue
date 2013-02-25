#!/usr/bin/env python
from __future__ import print_function
from distutils.core import setup, Command
from distutils.command.install_scripts import install_scripts
from glob import glob
import os
import sys
import platform

from setupext import (print_line, print_raw, print_status,
                      check_for_numpy, check_for_matplotlib,
                      check_for_qt4, check_for_ipython, check_for_scipy,
                      check_for_astropy, check_for_aplpy, check_for_pytest,
                      check_for_mock, check_for_pil, check_for_atpy,
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


class BuildQt(Command):

    user_options = [
        ('pyrcc4=', 'p', "Custom pyrcc4 command")
    ]

    def initialize_options(self):
        self.pyrcc4 = 'pyrcc4'

    def finalize_options(self):
        pass

    def run(self):

        from PyQt4.uic import compileUi

        for infile in glob(os.path.join('glue', 'qt', 'ui', '*.ui')):
            print("Compiling " + infile)
            directory, filename = os.path.split(infile)
            outfile = os.path.join(directory, filename.replace('.ui', '.py'))
            with open(outfile, 'wb') as out:
                compileUi(infile, out)

        import subprocess
        from shutil import copyfile

        print("Compiling glue/qt/glue.qrc")
        if sys.version_info[0] == 2:
            option = '-py2'
        else:
            option = '-py3'
        try:
            subprocess.call([self.pyrcc4, option, 'glue/qt/glue.qrc', '-o',
                             'glue/qt/glue_qt_resources.py'])
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

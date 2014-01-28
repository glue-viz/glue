#!/usr/bin/env python
"""
Guide users through installing Glue's dependencies
"""

#Unfortunately, we can't rely on setuptools' install_requires
#keyword, because matplotlib doesn't properly install its dependencies
from subprocess import check_call, CalledProcessError
import sys
from imp import find_module


class Dependency(object):
    def __init__(self, module, info, package=None, min_version=None):
        self.module = module
        self.info = info
        self.package = package or module
        self.min_version = min_version
        self.failed = False

    @property
    def installed(self):
        try:
            find_module(self.module)
            return True
        except ImportError:
            return False

    def install(self):
        if self.installed:
            return

        try:
            check_call(['pip', 'install', self.package])
        except CalledProcessError:
            self.failed = True

    def help(self):
        result = """
{module}:
******************

{info}

PIP package name:
{package}
""".format(module=self.module, info=self.info, package=self.package)
        return result

    def __str__(self):
        if self.installed:
            status = 'INSTALLED'
        elif self.failed:
            status = 'FAILED (%s)' % self.info
        else:
            status = 'MISSING (%s)' % self.info
        return "%20s:\t%s" % (self.module, status)


class QtDep(Dependency):
    def __init__(self):
        self.module = 'PyQt4 or PySide'
        self.info = ('GUI Library (install at http://bit.ly/YfTFxj or '
                     'http://bit.ly/Zci3Di)')
        self.package = 'N/A'

    @property
    def installed(self):
        for mod in ['PyQt4', 'PySide']:
            try:
                find_module(mod)
                return True
            except ImportError:
                pass
        else:
            return False

    def install(self):
        print ("*******************************\n"
               "CANNOT AUTOMATICALLY INSTALL PyQt4 or PySide.\n"
               "Install PyQt4 at http://bit.ly/YfTFxj, or\n"
               "Install PySide at http://bit.ly/Zci3Di\n"
               "*******************************\n"
               )


#Add any dependencies here
#Make sure to add new categories to the categories tuple
required = (
    QtDep(),
    Dependency('numpy', 'Required', min_version='1.4'),
    Dependency('matplotlib', 'Required for plotting', min_version='1.1'),
    Dependency('astropy', 'Used for FITS I/O, table reading, and WCS Parsing'))

general = (
    Dependency('dill', 'Used when saving Glue sessions'),
    Dependency('scipy', 'Used for some image processing calculation'),
    Dependency('skimage',
               'Used to read popular image formats (jpeg, png, etc.)',
               'scikit-image'))


ipython = (
    Dependency('IPython', 'Needed for interactive IPython terminal'),
    Dependency('pygments', 'Needed for interactive IPython terminal'),
    Dependency('zmq', 'Needed for interactive IPython terminal', 'pyzmq'))


astronomy = (
    Dependency('pyavm', 'Used to parse AVM metadata in image files', 'PyAVM'),)


testing = (
    Dependency('mock', 'Used in test code'),
    Dependency('pytest', 'Used in test code'))


categories = (('required', required),
              ('general', general),
              ('ipython terminal', ipython),
              ('astronomy', astronomy),
              ('testing', testing))

dependencies = {d.module: d for c in categories for d in c[1]}


def show_status():
    for category, deps in categories:
        print "%21s" % category.upper()
        for dep in deps:
            print dep
        print '\n'


def install_all():
    for category, deps in categories:
        for dep in deps:
            dep.install()


def install_selected(modules):
    modules = set(m.lower() for m in modules)

    for category, deps in categories:
        for dep in deps:
            if dep.installed:
                continue
            if dep.module.lower() in modules or category.lower() in modules:
                dep.install()


def main(argv=None):
    argv = argv or sys.argv

    usage = """usage:
    #install all dependencies
    %s install

    #show all dependencies
    %s list

    #install a specific dependency or category
    %s install astropy
    %s install astronomy

    #display information about a dependency
    %s info astropy
""" % ('glue-deps', 'glue-deps', 'glue-deps', 'glue-deps', 'glue-deps')

    if len(argv) < 2 or argv[1] not in ['install', 'list', 'info']:
        sys.stderr.write(usage)
        sys.exit(1)

    if argv[1] == 'info':
        if len(argv) != 3:
            sys.stderr.write(usage)
            sys.stderr.write("Please specify a dependency\n")
            sys.exit(1)

        dep = dependencies.get(argv[2], None)

        if dep is None:
            sys.stderr.write("Unrecognized dependency: %s\n" % argv[2])
            sys.exit(1)

        print dep.help()
        sys.exit(0)

    if argv[1] == 'list':
        show_status()
        sys.exit(0)

    # argv[1] == 'install'
    if len(argv) == 2:
        install_all()
        show_status()
        sys.exit(0)

    install_selected(argv[2:])
    show_status()


if __name__ == "__main__":
    main()

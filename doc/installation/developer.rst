Installing the latest developer version
=======================================

If you are using conda, the easiest way to get all the dependencies installed
when using the developer version is to first install the stable version, which
will pull in all the dependencies, then to remove it and install the developer
version::

    conda install -c conda-forge glueviz glue-vispy-viewers
    conda remove glueviz glue-vispy-viewers

    git clone git://github.com/glue-viz/glue
    cd glue
    pip install .
    cd ..

    git clone git://github.com/glue-viz/glue-vispy-viewers
    cd glue-vispy-viewers
    pip install .
    cd ..

If you want to then uninstall the developer versions and install the stable
versions again, you can uninstall the developer versions with::

    pip uninstall glueviz glue-vispy-viewers

then install the stable versions with conda.

.. _configuration:

Configuring Glue via a startup file
===================================

Glue uses a configuration system to customize aspects such as which
visualization modules it loads, what link functions to use, etc. This
allows users who create their own glue modules to easily incorporate
them into the main GUI environment.

The glue configuration file is called ``config.py``. Glue looks for this file
in the following locations, in order:

 * The current working directory
 * The path specified in the ``GLUERC`` environment variable, if present
 * The path ``.glue/config.py`` within the user's home directory

To obtain a fresh ``config.py`` file to edit, run the command line program::

   glue-config

Which will create a new file at ``~/.glue/config.py``

Example Usage: Custom Link Functions
------------------------------------

As an example, let's create some translation functions which will allow us to
convert temperatures in Celsius to Farenheit::

    from glue.config import link_function

    @link_function(info="Celsius to Fahrenheit", output_labels=['F'])
    def celsius2farhenheit(c):
        return c  * 9. / 5. + 32

    @link_function(info="Fahrenheit to Celsius", output_labels=['C'])
    def farhenheit2celsius(f):
        return (f - 32) * 5. / 9.

More details about this are provided in :ref:`customization`, but for now, let's just assume this is how we make custom linking functions. We can copy this code into ``~/.glue/config.py`` file. Next time we start up Glue, the link functions now appear in the Link Dialog:

.. image:: images/link_functions.png
   :align: center



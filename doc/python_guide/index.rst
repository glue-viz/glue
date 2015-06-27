Python Guide
------------

Glue is designed with "data-hacking" workflows in mind. Because Glue
is a python library for data interaction, it blurs the boundary between
GUI-centric and code-centric data exploration.

There are many ways to leverage Glue from python. Among other
things, you can write code to do the following:

#. :ref:`Send data <qglue>` in the form of NumPy arrays or Pandas DataFrames to Glue for exploration
#. Write :ref:`startup scripts <startup_scripts>` that automatically load and clean data, before starting Glue.
#. Write custom functions to :ref:`parse files <custom_data_factory>`, and plug these functions into the Glue GUI.
#. Write custom functions to :ref:`link datasets <custom_links>`, and plug these into the Glue GUI.
#. Create your own visualization modules.


The following pages discuss these concepts. You may want to start by reading the :ref:`Glue Data guide <data_tutorial>` for a description of how Glue organizes data.

.. toctree::
   :maxdepth: 2

   data_tutorial.rst
   glue_from_python.rst
   data_viewer_options.rst
   liveupdate.rst

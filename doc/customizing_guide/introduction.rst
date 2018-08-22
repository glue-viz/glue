Introduction to customizing/extending glue
==========================================

Glue has been designed from the ground up to be easily customizable by users,
with a number of customization aspects requiring only simple Python code and no
knowledge of e.g. Qt. There are two main ways to customize/extend glue:

* Installing an existing plugin package that will automatically extend glue.
  For example, you can install a package called glue-geospatial to add the
  ability to read in geospatial (e.g. GeoTIFF) data, or glue-medical to add
  the ability to read in common medical (e.g. DICOM) formats. You can find
  a list of available plugins as well as information on installing them
  at :ref:`available_plugins`

* Writing your own customizations and placing these in a file named
  ``config.py`` (we will look shortly at where to place that file). This is
  ideal if you just want to customize glue quickly for your own work, or if you
  are prototyping ideas for a more advanced plugin package. To find out more
  about what aspects of glue you can customize and how to do it, you can read
  :ref:`configuration`.

If you are developing something for your own work only, then using the
``config.py`` approach is usually sufficient, and you don't have to develop a
fully fledged plugin package. But if you are interested in distributing your
plugin for others to use, then you should make sure you read
:ref:`writing_plugin` for more information on how to do this.

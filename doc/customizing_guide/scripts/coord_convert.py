from glue.config import link_function


@link_function(info="Celsius to Fahrenheit",
               output_labels=['F'])
def celsius2farhenheit(c):
    return c * 9. / 5. + 32


@link_function(info="Fahrenheit to Celsius",
               output_labels=['C'])
def farhenheit2celsius(f):
    return (f - 32) * 5. / 9.

from glue.config import link_function

@link_function(info="Link from deg to rad", output_labels=['rad'])
def deg_to_rad(deg):
    return deg * 3.14159 / 180

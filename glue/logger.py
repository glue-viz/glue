from logging import getLogger, StreamHandler
logger = getLogger("glue")
logger.addHandler(StreamHandler())

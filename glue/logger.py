from logging import getLogger, basicConfig, NullHandler
basicConfig()
logger = getLogger("glue")
# Default to Null unless we override this later
logger.addHandler(NullHandler())

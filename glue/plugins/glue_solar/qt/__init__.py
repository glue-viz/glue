from .splash_screen import SunpyQtSplashScreen


# def setup():
#     from glue.config import qt_client
#     qt_client.add(SunpyQtSplashScreen)


def get_splash():
    """Instantiate a splash screen"""
    # from glue.app.qt.splash_screen import QtSplashScreen
    splash = SunpyQtSplashScreen()
    return splash

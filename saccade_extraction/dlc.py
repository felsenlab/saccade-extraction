try:
    import deeplabcut as dlc
except:
    dlc = None
import logging
import os
import contextlib

_DLC_CONFIG_FILE = None

def setConfigFile(configFile):
    """
    """

    global _DLC_CONFIG_FILE
    _DLC_CONFIG_FILE = configFile

    return

def getConfigFile():
    """
    """

    return _DLC_CONFIG_FILE

def analyzeVideosQuietly(*args, **kwargs):
    """
    Call DeepLabCut's analyze_videos function but suppress messaging
    """

    if dlc is None:
        raise Exception('DeepLabCut import failed')

    kwargs_ = {}
    kwargs_.update(kwargs)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        logging.disable(logging.CRITICAL)  # Disable logging
        try:
            return dlc.analyze_videos(*args, **kwargs_)
        finally:
            logging.disable(logging.NOTSET)  # Re-enable logging
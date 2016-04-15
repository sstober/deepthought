__author__ = 'sstober'

import logging
from pylearn2.utils.logger import CustomFormatter, CustomStreamHandler

def configure_custom(debug=False, stdout=None, stderr=None):
    """
    copied from pylearn2.utils.logger to obtain similar behavior for deepthought extensions

    Configure the logging module to output logging messages to the
    console via `stdout` and `stderr`.

    Parameters
    ----------
    debug : bool
        If `True`, display DEBUG messages on `stdout` along with
        INFO-level messages.
    stdout : file-like object, optional
        Stream to which DEBUG and INFO messages should be written.
        If `None`, `sys.stdout` will be used.
    stderr : file-like object, optional
        Stream to which WARNING, ERROR, CRITICAL messages will be
        written. If `None`, `sys.stderr` will be used.

    Notes
    -----
    This uses `CustomStreamHandler` defined in this module to
    set up a console logger. By default, messages are formatted
    as "LEVEL: message", where "LEVEL:" is omitted if the
    level is INFO.

    WARNING, ERROR and CRITICAL level messages are logged to
    `stderr` (or the provided substitute)

    N.B. it is **not** recommended to pass `sys.stdout` or
    `sys.stderr` as constructor arguments explicitly, as certain
    things (like nosetests) can reassign these during code
    execution! Instead, simply pass `None`.
    """
    top_level_logger = logging.getLogger(__name__.split('.')[0])

    # Do not propagate messages to the root logger.
    top_level_logger.propagate = False

    # Set the log level of our logger, either to DEBUG or INFO.
    top_level_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Get rid of any extant logging handlers that are installed.
    # This means we can call configure_custom() more than once
    # and have it be idempotent.
    while top_level_logger.handlers:
        top_level_logger.handlers.pop()

    # Install our custom-configured handler and formatter.
    fmt = CustomFormatter()
    handler = CustomStreamHandler(stdout=stdout, stderr=stderr, formatter=fmt)
    top_level_logger.addHandler(handler)
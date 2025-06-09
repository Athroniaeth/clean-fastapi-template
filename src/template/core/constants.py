from enum import StrEnum


class Level(StrEnum):
    """
    Log levels used to trace application execution.

    Attributes:
        TRACE   : Very fine-grained details for deep debugging.
        DEBUG   : Debugging information for developers.
        INFO    : General operational events.
        WARNING : Unexpected behavior that doesn't stop execution.
        ERROR   : Critical issues that affect functionality.
    """

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

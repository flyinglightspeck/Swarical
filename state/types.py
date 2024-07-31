from enum import Enum


class StateTypes(Enum):
    AVAILABLE = 1
    BUSY_ANCHOR = 2
    BUSY_LOCALIZING = 3
    WAITING = 4
    DEPLOYING = 5

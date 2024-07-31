from enum import Enum


class MessageTypes(Enum):
    STOP = 0
    DUMMY = 99
    CHALLENGE_INIT = 1
    CHALLENGE_ACCEPT = 2
    CHALLENGE_ACK = 3
    CHALLENGE_FIN = 4
    FOLLOW = 5
    MERGE = 6
    FOLLOW_MERGE = 7
    SET_AVAILABLE = 8
    SET_WAITING = 9
    LEASE_GRANT = 10
    LEASE_RENEW = 11
    SIZE_QUERY = 12
    SIZE_REPLY = 13
    THAW_SWARM = 14
    REPORT = 15
    FIN = 16
    LEASE_CANCEL = 17

    RENEW_LEASE_INTERNAL = 20
    SET_AVAILABLE_INTERNAL = 21
    FAIL_INTERNAL = 22
    THAW_SWARM_INTERNAL = 23

    QUERY_SWARM = 30
    REPLY_SWARM = 31

    GOSSIP = 40
    GOSSIP_INTERNAL = 41
    NOTIFY = 42
    NOTIFY_INTERNAL = 43

    UN_ANCHOR = 50
    UN_ANCHOR_INTERNAL = 51

    QUERY_SWEET = 60
    REPLY_SWEET = 61
    UPDATE_GRAPH = 62

    def get_cat(self):
        if 1 <= self.value <= 4:
            return 'CHALLENGE'
        elif 5 <= self.value <= 7:
            return 'FOLLOW'
        elif 8 <= self.value <= 9:
            return 'STATE_CHANGE'
        elif 10 <= self.value <= 11 or self.value == 17:
            return 'LEASE'
        elif 12 <= self.value <= 13:
            return 'SIZE'
        elif self.value == 14:
            return 'THAW'
        elif self.value == 0 or self.value == 16:
            return 'TERMINATION'

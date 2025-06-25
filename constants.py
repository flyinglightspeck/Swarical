class Constants:
    BROADCAST_PORT = 5000
    SERVER_PORT = 6000
    WORKER_ADDRESS = ("", 5000)

    PLATFORM = 'local'  # Set to local, cloudlab, or aws

    if PLATFORM == 'local':
        SERVER_ADDRESS = ("localhost", 6000)  # localhost
        BROADCAST_ADDRESS = ("<broadcast>", 5000)  # localhost
    elif PLATFORM == 'cloudlab':
        SERVER_ADDRESS = ("10.0.1.1", 6000)  # cloudlab
        BROADCAST_ADDRESS = ("10.0.1.255", 5000)  # cloudlab
    elif PLATFORM == 'aws':
        SERVER_ADDRESS = ("172.31.93.249", 6000)  # aws

    MULTICAST_GROUP_ADDRESS = ('224.3.29.25', 5000)
    MULTICAST_GROUP = '224.3.29.25'


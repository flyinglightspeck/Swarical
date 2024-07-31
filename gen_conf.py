import itertools
import os

def_conf = {
    "GOSSIP_TIMEOUT": "5",
    "GOSSIP_SWARM_COUNT_THRESHOLD": "3",
    "THAW_SWARMS": "False",
    "INITIAL_RANGE": "200",
    "MAX_RANGE": "200",
    "DROP_PROB_SENDER": "0",
    "DROP_PROB_RECEIVER": "0",
    "STATE_TIMEOUT": "0.5",
    "SIZE_QUERY_TIMEOUT": "10",
    "DEAD_RECKONING_ANGLE": "5",
    "CHALLENGE_PROB_DECAY": "1.25",
    "INITIAL_CHALLENGE_PROB": "1",
    "CHALLENGE_LEASE_DURATION": "0.25",
    "CHALLENGE_ACCEPT_DURATION": "0.02",
    "CHALLENGE_INIT_DURATION": "0",
    "FAILURE_TIMEOUT": "0",
    "FAILURE_PROB": "0",
    "NUMBER_ROUND": "5",
    "ACCELERATION": "10",
    "DECELERATION": "10",
    "MAX_SPEED": "10",
    "DISPLAY_CELL_SIZE": "0.05",
    "HD_TIMOUT": "5",
    "SIZE_QUERY_PARTICIPATION_PERCENT": "1",
    "DECENTRALIZED_SWARM_SIZE": "False",
    "CENTRALIZED_SWARM_SIZE": "False",
    "PROBABILISTIC_ROUND": "False",
    "CENTRALIZED_ROUND": "True",
    "BUSY_WAITING": "False",
    "MIN_ADJUSTMENT": "0",
    "SAMPLE_SIZE": "0",
    "DURATION": "60",
    "SHAPE": "'outring'",
    "RESULTS_PATH": "'results'",
    "MULTICAST": "True",
    "THAW_MIN_NUM_SWARMS": "1",
    "THAW_PERCENTAGE_LARGEST_SWARM": "80",
    "THAW_INTERVAL": "1",
    "CAMERA": "'w'",
    "SCALE": "1",
    "SS_ERROR_MODEL": "1",
    "SS_ERROR_PERCENTAGE": "0.0",
    "SS_ACCURACY_PROBABILITY": "0.0",
    "SS_NUM_SAMPLES": "1",
    "SS_SAMPLE_DELAY": "0",
    "STANDBY": "False",
    "GROUP": "False",
    "GROUP_TYPE": "'spanning_3'",
    "MULTIPLE_ANCHORS": "True",
    "FILE_NAME_KEYS": "[('GROUP_TYPE', 'V'), ('DEAD_RECKONING_ANGLE', 'D'), ('CAMERA', 'C'), ('SS_ERROR_MODEL', 'M')]",
    "DIR_KEYS": "[]",
}

props = [
    {
        "keys": ["CAMERA", "SS_ERROR_MODEL"],
        "values": [
            {"CAMERA": "'w'", "SS_ERROR_MODEL": "1"},
            # {"CAMERA": "'r'", "SS_ERROR_MODEL": "1"},
            # {"CAMERA": "'w'", "SS_ERROR_MODEL": "0"},
        ],
    },
    {
        "keys": ["SHAPE", "GROUP_TYPE", "SCALE"],
        "values": [
            # {"SHAPE": "'chess_408_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'chess_408_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'chess_408_50_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'chess_408_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'chess_408_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'palm_725_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'palm_725_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'palm_725_50_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'palm_725_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'palm_725_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'kangaroo_972_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'kangaroo_972_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'kangaroo_972_50_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'kangaroo_972_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'kangaroo_972_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'dragon_1147_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'dragon_1147_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'dragon_1147_50_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'dragon_1147_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'dragon_1147_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'skateboard_1372_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'skateboard_1372_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'skateboard_1372_50_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'skateboard_1372_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},
            # {"SHAPE": "'skateboard_1372_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v3'"},

            # {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v3'"},

            # {"SHAPE": "'skateboard_1372_5_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_10_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_150_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_200_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'"},

            # {"SHAPE": "'chess_408_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},

            # {"SHAPE": "'chess_100_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'chess_100_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            {"SHAPE": "'chess_100_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "2.3"},

            # {"SHAPE": "'chess_408_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'chess_408_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            {"SHAPE": "'chess_408_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            # {"SHAPE": "'chess_408_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'chess_408_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'palm_725_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'palm_725_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            {"SHAPE": "'palm_725_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            # {"SHAPE": "'palm_725_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'palm_725_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'kangaroo_972_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'kangaroo_972_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            {"SHAPE": "'kangaroo_972_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            # {"SHAPE": "'kangaroo_972_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'kangaroo_972_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'dragon_1147_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'dragon_1147_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            {"SHAPE": "'dragon_1147_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "4"},
            # {"SHAPE": "'dragon_1147_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'dragon_1147_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},


            {"SHAPE": "'skateboard_1372_5_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_10_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_150_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_200_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},

            {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v3'", "SCALE": "3.4"},

            # {"SHAPE": "'skateboard_1372_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1372_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},

            # {"SHAPE": "'skateboard_1855_5_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1855_10_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1855_50_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1855_150_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            # {"SHAPE": "'skateboard_1855_200_spanning_2'", "GROUP_TYPE": "'spanning_2_v2'"},
            #
            # {"SHAPE": "'chess_408_5_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'chess_408_10_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'chess_408_50_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'chess_408_150_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'chess_408_200_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'palm_725_5_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'palm_725_10_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'palm_725_50_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'palm_725_150_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'palm_725_200_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'kangaroo_972_5_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'kangaroo_972_10_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'kangaroo_972_50_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'kangaroo_972_150_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'kangaroo_972_200_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'dragon_1147_5_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'dragon_1147_10_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'dragon_1147_50_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'dragon_1147_150_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'dragon_1147_200_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'skateboard_1372_5_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'skateboard_1372_10_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'skateboard_1372_50_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'skateboard_1372_150_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'skateboard_1372_200_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'racecar_3720_5_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'racecar_3720_10_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'racecar_3720_50_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'racecar_3720_150_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'racecar_3720_200_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            #
            # {"SHAPE": "'chess_408_mst'", "GROUP_TYPE": "'universal'"},
            # {"SHAPE": "'palm_725_mst'", "GROUP_TYPE": "'universal'"},
            # {"SHAPE": "'kangaroo_972_mst'", "GROUP_TYPE": "'universal'"},
            # {"SHAPE": "'dragon_1147_mst'", "GROUP_TYPE": "'universal'"},
            # {"SHAPE": "'skateboard_1372_mst'", "GROUP_TYPE": "'universal'"},

            # {"SHAPE": "'racecar_3720_mst'", "GROUP_TYPE": "'universal'"},

            # {"SHAPE": "'chess_100_mst'", "GROUP_TYPE": "'universal'"},
            # {"SHAPE": "'chess_408_mst'", "GROUP_TYPE": "'universal'"},
            # {"SHAPE": "'chess_100_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'chess_100_spanning_3'", "GROUP_TYPE": "'spanning_3'"},
            # {"SHAPE": "'chess_408_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'chess_408_spanning_3'", "GROUP_TYPE": "'spanning_3'"},
            # {"SHAPE": "'grid_144_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'grid_225_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'grid_324_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'grid_400_spanning_2'", "GROUP_TYPE": "'spanning_2'"},
            # {"SHAPE": "'grid_100_spanning'", "GROUP_TYPE": "'spanning'"},
            # {"SHAPE": "'grid_144_spanning'", "GROUP_TYPE": "'spanning'"},
            # {"SHAPE": "'grid_225_spanning'", "GROUP_TYPE": "'spanning'"},
            # {"SHAPE": "'grid_324_spanning'", "GROUP_TYPE": "'spanning'"},
            # {"SHAPE": "'grid_400_spanning'", "GROUP_TYPE": "'spanning'"},
            # "'grid_36_spanning'",
            # "'grid_100_spanning_2'",
            # "'grid_400_spanning_2'",
            # "'chess_408_spanning'",
            # "'chess_spanning_2'",
            # "'chess_100_spanning_3'",
            # "'chess_408_spanning_3'",
            # "'palm_725_spanning_3'",
            # "'dragon_1147_spanning_3'",
            # "'skateboard_1372_spanning_3'",
        ]
    },
    {
        "keys": ["DEAD_RECKONING_ANGLE", "SS_ERROR_PERCENTAGE"],
        "values": [
            {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.0"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.0"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.01"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.1"},

            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'10'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'50'"},
            #
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.05", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.05", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'50'"},
            #
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "3", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "3", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'50'"},
            #
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "3", "SS_ERROR_PERCENTAGE": "0.05", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "3", "SS_ERROR_PERCENTAGE": "0.05", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'50'"},
            #
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'10'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.01", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'10'"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.1", "SHAPE": "'50'"},
            #
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'10'"},
            # {"DEAD_RECKONING_ANGLE": "1", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'50'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'2'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'10'"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0", "SHAPE": "'50'"},
        ]
    },
]

if __name__ == '__main__':
    file_name = "config"
    class_name = "Config"

    props_values = [p["values"] for p in props]
    print(props_values)
    combinations = list(itertools.product(*props_values))
    print(len(combinations))

    if not os.path.exists('experiments'):
        os.makedirs('experiments', exist_ok=True)

    for j in range(len(combinations)):
        c = combinations[j]
        conf = def_conf.copy()
        for i in range(len(c)):
            for k in props[i]["keys"]:
                if isinstance(c[i], dict):
                    conf[k] = c[i][k]
                else:
                    conf[k] = c[i]
        with open(f'experiments/{file_name}{j}.py', 'w') as f:
            f.write(f'class {class_name}:\n')
            for key, val in conf.items():
                f.write(f'    {key} = {val}\n')

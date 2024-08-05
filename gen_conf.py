import itertools
import os

def_conf = {
    "INITIAL_RANGE": "1000",
    "MAX_RANGE": "1000",
    "DROP_PROB_SENDER": "0",
    "DROP_PROB_RECEIVER": "0",
    "STATE_TIMEOUT": "0.5",
    "DEAD_RECKONING_ANGLE": "5",
    "ACCELERATION": "10",
    "DECELERATION": "10",
    "MAX_SPEED": "10",
    "DISPLAY_CELL_SIZE": "0.05",
    "BUSY_WAITING": "False",
    "DURATION": "60",
    "SHAPE": "'grid_36_spanning_2'",
    "RESULTS_PATH": "'results'",
    "MULTICAST": "True",
    "CAMERA": "'w'",
    "SCALE": "1",
    "SS_ERROR_MODEL": "0",
    "SS_ERROR_PERCENTAGE": "0.0",
    "SS_ACCURACY_PROBABILITY": "0.0",
    "SS_NUM_SAMPLES": "1",
    "SS_SAMPLE_DELAY": "0",
    "STANDBY": "False",
    "GROUP": "False",
    "SWEET_RANGE": "(6, 8)",
    "GROUP_TYPE": "'spanning_3'",
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
            # {"SHAPE": "'chess_408_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'chess_100_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "2.3"},
            {"SHAPE": "'chess_408_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'palm_725_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'kangaroo_972_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'dragon_1147_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "4"},


            {"SHAPE": "'skateboard_1372_5_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_10_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_150_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_200_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v2'", "SCALE": "3.4"},

            {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2'", "SCALE": "3.4"},
            {"SHAPE": "'skateboard_1372_50_spanning_2_sb'", "GROUP_TYPE": "'spanning_2_v3'", "SCALE": "3.4"},
        ]
    },
    {
        "keys": ["DEAD_RECKONING_ANGLE", "SS_ERROR_PERCENTAGE"],
        "values": [
            {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.0"},
            # {"DEAD_RECKONING_ANGLE": "0", "SS_ERROR_PERCENTAGE": "0.0"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.01"},
            # {"DEAD_RECKONING_ANGLE": "5", "SS_ERROR_PERCENTAGE": "0.1"},
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

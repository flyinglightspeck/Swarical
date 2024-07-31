#!/bin/bash

source aws_vars.sh

python3 gen_conf.py

for (( i=1; i<N; i++ )); do
    server_addr=${USERNAME}@${HOSTNAMES[$i]}
    ssh -oStrictHostKeyChecking=no -i ${LOCAL_KEY_PATH} "${server_addr}" "cd ${REPO_PATH} && python3.9 gen_conf.py" &
done

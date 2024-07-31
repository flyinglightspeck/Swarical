#!/bin/bash

source cloudlab_vars.sh

python3 gen_conf.py

for (( i=1; i<N; i++ )); do
    server_addr=${USERNAME}@node-$i.${HOSTNAME}
    ssh -oStrictHostKeyChecking=no -f "${server_addr}" "cd ${REPO_PATH} && python3.9 gen_conf.py" &
done

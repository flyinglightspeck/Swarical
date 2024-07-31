#!/bin/bash

source cloudlab_vars.sh

now=$(date +%d_%b_%H_%M_%S)
port="6000"

for (( i=1; i<N; i++ )); do
    server_addr=${USERNAME}@node-$i.${HOSTNAME}
    ssh -o ConnectionAttempts=5 -oStrictHostKeyChecking=no -f "${server_addr}" "ulimit -n 9999 && cd ${REPO_PATH} && cp experiments/config$1.py config.py && sleep 10 && sudo nohup python3.9 server.py ${N} ${i} ${now} ${port} > my.log 2>&1 &" &
done

ulimit -n 9999 && cp "experiments/config$1.py" config.py && sleep 1 && sudo python3.9 server.py "${N}" 0 "${now}" "${port}"

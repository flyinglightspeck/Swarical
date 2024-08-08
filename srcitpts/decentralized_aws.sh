#!/bin/bash

source ../aws_local_vars.sh

setup_nodes=false
kill_processes=false
update_repo=false
copy_key=false
delete_results=false
run_nohup=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            setup_nodes=true
            ;;
        --kill)
            kill_processes=true
            ;;
        --update)
            update_repo=true
            ;;
        --copy-key)
            copy_key=true
            ;;
        --delete-results)
            delete_results=true
            ;;
        --run-nohup)
            run_nohup=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$setup_nodes" = true ]; then
for (( i=0; i<N; i++ )); do
    server_addr=${USERNAME}@${HOSTNAMES[$i]}
    ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "git clone ${GITHUB_REPO} && cd Swarical && bash setup.sh" &
done
fi

if [ "$copy_key" = true ]; then
    scp -oStrictHostKeyChecking=no -i ${KEY_PATH} ${KEY_PATH} "${USERNAME}@${HOSTNAMES[0]}:Swarical"
fi


for (( i=0; i<N; i++ )); do
    server_addr=${USERNAME}@${HOSTNAMES[$i]}

    if [ "$update_repo" = true ]; then
        ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "cd Swarical && git pull" &
    fi

    # clean up results
    if [ "$delete_results" = true ]; then
        ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "rm Swarical/result${i}.tar.gz" &
        ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "rm -rf Swarical/experiments" &
        ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "rm -rf Swarical/results" &
    fi

    # kill processes
    if [ "$kill_processes" = true ]; then
        ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "pkill python3.9" &
    fi
done

# Run nohup script on the primary node
if [ "$run_nohup" = true ]; then
    server_addr=${USERNAME}@${HOSTNAMES[0]}
    ssh -oStrictHostKeyChecking=no ${server_addr} "cd Swarical && bash nohup_aws_run.sh"
fi

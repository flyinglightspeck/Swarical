#!/bin/bash

source ../aws_local_vars.sh


download_results=false
download_logs=false
extract_results=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results)
            download_results=true
            ;;
        --logs)
            download_logs=true
            ;;
        --extract-results)
            extract_results=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done


# specify the destination directory to download the results to
dir_name="results/test"


mkdir -p "${dir_name}"

for (( i=0; i<N; i++ )); do
    server_addr=${USERNAME}@${HOSTNAMES[$i]}
    # download results
    if [ "$download_results" = true ]; then
        ssh -oStrictHostKeyChecking=no -i ${KEY_PATH} "${server_addr}" "cd Swarical && tar czf result${i}.tar.gz results" && scp -i ${KEY_PATH} "${server_addr}":Swarical/result"${i}".tar.gz "${dir_name}" &
    fi

    # download logs
    if [ "$download_logs" = true ]; then
        scp -i ${KEY_PATH} "${server_addr}":Swarical/my.log "${dir_name}/my${i}.log" &
    fi
done


if [ "$extract_results" = true ]; then
    for (( j=0; j<N; j++ )); do
        tar -xf "${dir_name}/result${j}.tar.gz" -C "${dir_name}" &
    done
fi

#!/bin/bash

N=10 # number of total servers
USERNAME="ubuntu"
KEY_PATH="~/path/to/key.pem"
REPO_PATH="Swarical"
GITHUB_REPO="https://github.com/flyinglightspeck/Swarical.git"
now=$(date +%d-%b-%H_%M_%S)

HOSTNAMES=(
"list of public ips of servers"
)

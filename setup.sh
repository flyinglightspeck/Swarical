#!/bin/bash
sudo apt update -y
if ! command -v pip3 &> /dev/null
sudo apt install python3.9 -y
then
    echo "pip3 could not be found"
    echo "installing pip3 ..."
    sudo apt install python3-pip -y
fi
python3.9 -m pip install --upgrade pip
python3.9 -m pip install -r requirements.txt
echo "now run python3.9 server.py"
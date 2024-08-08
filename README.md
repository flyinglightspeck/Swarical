# Swarical
Swarical uses the hardware specification of sensors mounted on FLSs to convert mesh files into point clouds that enable a swarm of FLSs to localize at the highest accuracy afforded by their hardware.

Authors:  Hamed Alimohammadzadeh(halimoha@usc.edu) and Shahram Ghandeharizadeh (shahram@usc.edu)

## Features

  * Three decentralized algorithms that localize FLSs to illuminate a 3D or 2D point cloud.
  * A state machine that implements a decentralized algorithm.
  * A planner that creates Swarm-Tree and FLS-Trees using K-Means and MST.
  * Launches multiple processes, one process per Flying Light Speck, FLS.  With large point clouds (FLSs), the software scales to utilize multiple servers. Both CloudLab and Amazon AWS are supported.


## Limitations
  * With large point clouds and the Linux operating system, the execution of the software may exhaust the max open files supported by the operating system.  See below, Error with Large Point Clouds, for how to resolve. 


## Clone
```
git clone https://github.com/flyinglightspeck/Swarical.git
```


## Running on a Laptop or a Server

This software was implemented and tested using Python 3.9.0.

We recommend using PyCharm, which enables the software to run across multiple operating systems, e.g., Windows, MacOS, etc.

### Running using a (PyCharm) Terminal

Run ``bash setup.sh`` to install the requirements.

The variables specified in `config.py` control settings.  

If running on a laptop/server with a limited number of cores, use a point cloud with a few points (e.g., grid_36_spanning_2).  As a general guideline, their values should not exceed four times the number of physical cores.

This program is designed to scale horizontally across multiple servers and run with large point clouds. Each point is assigned to a Flying Light Speck, a process launched by this program.  

Run `server.py` after adjusting the settings of `config.py` (see below). 

### Running using virtual environment, Venv

You can create and activate a virtual environment by following these steps.
First, you'll need to create a virtual environment using Venv. You can use any name instead of env.

```
cd Swarical
python3.9 -m venv env
```

Then, activate the virtual environment.

```
source env/bin/activate
```

On Windows use the following instead:

```
env/Scripts/activate.bat //In CMD
env/Scripts/Activate.ps1 //In Powershel
```

Install the requirements:

```
pip3 install -r requirements.txt
```

You can now run `server.py`. Finally, the virtual environment can be deactivated by running `deactivate` in the terminal.


### A Point Cloud
We provide several point clouds, e.g., a Chess piece.  The value of variable SHAPE in config.py controls the used point cloud.  Set the `SHAPE` value to the shape name (use the file name of .txt files in the `assets` directory as the value of the `SHAPE`, e.g., `dragon_1147_50_spanning_2_sb`).  The repository comes with the following shapes: `chess`, `dragon`, `kangaroo`, `racecar`, `skateboard`, `grid_36`.

The file name parts separated by '_' specifies the shape name, number of points, group size, and the planner variant.

## Running on Multiple Servers: Amazon AWS
First, set up a cluster of servers. Ideally, the total number of cores of the servers should equal or be greater than the number of points in the point cloud (number of FLSs).

Set up a multicast domain (For more information on how to create a multicast domain, see aws docs: https://docs.aws.amazon.com/vpc/latest/tgw/manage-domain.html)

Add your instances to the multicast domain. Use the value of MULTICAST_GROUP_ADDRESS in the constants.py for the group address.

Ensure you allow all UDP, TCP, and IGMP(2) traffic in your security group.

After setting up AWS:

Choose one of the instances as the primary instance.

Set the private IP address of the primary instance as the `SERVER_ADDRESS` in `constants.py`.

In `aws_vars.sh`, set `N` to the number of total instances you have. Set the `KEY_PATH` as the path to the AWS key pair on your machine. List the private IP addresses of all the instances in `HOSTNAMES`; the primary should be the first.

In `aws_local_vars.sh`, set `N` to the number of total instances you have. Set the `LOCAL_KEY_PATH` as the path to the AWS key pair on the primary instance. List the public IP addresses of all the instances in `HOSTNAMES`; the primary should be the first.

Configure the experiment(s) you want to run by modifying `gen_conf.py`.

Clone the repository and set up the project by running `setup.sh` on each server using the following. Then copy the AWS key to the primary instance.

```
bash scripts/decentralized_aws.sh --setup
bash scripts/decentralized_aws.sh --copy-key
```

Finally, the experiments will be started by running nohup_run.sh on the primary instance.

```
bash scripts/decentralized_aws.sh --run-nohup
```

After the experiments are finished, you can download the results using `scripts/download_aws.sh`
```
bash scripts/download_aws.sh --results
bash scripts/download_aws.sh --extract-results
```

Finally use the `utils/file.py` to post-process the results to generate charts.



## Error with Large Point Clouds
If you encountered an error regarding not enough fds, increase max open files system-wide to be able to run a large point cloud:

``sudo vim /etc/sysctl.conf``

Add the following line:

``fs.file-max = 9999``

``sudo sysctl -p``

Reload terminal and then run this command:

``ulimit -n 9999``

## Online Localization: ISR, HC, and RSF
We present three localization techniques. The main difference between the techniques is the amount of concurrent movements by the FLSs. ISR is superior to HC and RSF. It is faster and more accurate than the other, minimizing the total distance traveled by FLSs. The use of RSF is not recommended as it fails to localize large point clouds effectively. Compare the performance of the techniques by watching the demonstrations below:
* [Inter-Swarm Rounds, ISR](https://youtu.be/GncnoqqYT_w)
* [Highly Concurrent, HC](https://youtu.be/0_Gs7IkDADw)
* [Rounds across the Swarm-tree and FLS-trees, RSF](https://youtu.be/YlLCxW32tvg)

## Citations

Hamed Alimohammadzadeh, and Shahram Ghandeharizadeh. 2024. Swarical: An Integrated Hierarchical Approach to Localizing Flying Light Specks. In Proceedings of the 32nd ACM International Conference on Multimedia (MM '24). Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3664647.3681080

BibTex:
```
@inproceedings{10.1145/3664647.3681080, 
author = {Alimohammadzadeh, Hamed and Ghandeharizadeh, Shahram}, 
title = {Swarical: An Integrated Hierarchical Approach to Localizing Flying Light Specks}, 
year = {2024}, 
isbn = {9798400706868}, 
publisher = {Association for Computing Machinery}, 
address = {New York, NY, USA}, 
url = {https://doi.org/10.1145/3664647.3681080}, 
doi = {10.1145/3664647.3681080}, 
abstract = {Swarical, a \underline{Swar}m-based hierarch\underline{ical} localization technique, enables miniature drones, known as Flying Light Specks (FLSs), to accurately and efficiently localize and illuminate complex 2D and 3D shapes. Its accuracy depends on the physical hardware (sensors) of FLSs, which are used to track neighboring FLSs in order to localize themselves. It uses the hardware specification to convert mesh files into point clouds that enable a swarm of FLSs to localize at the highest accuracy afforded by their hardware. Swarical considers a heterogeneous mix of FLSs with different orientations for their tracking sensors, ensuring a line of sight between a localizing FLS and its anchor FLS. We present an implementation using Raspberry cameras and ArUco markers. A comparison of Swarical with a state of the art decentralized localization technique shows that it is as accurate and more than 2x faster.}, 
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia}, 
location = {Melbourne, VIC, Australia}, 
series = {MM '24} 
}
```

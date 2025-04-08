# Running on Multiple Servers: CloudLab

## Create Cloudlab Experiment

First, set up a cluster of servers. Creat a cloudlab profile using the file profile.py, then start your own experiments
with any chosen type of nodes. Ideally, the total number of cores of the servers should equal or be greater than the
number of points in the point cloud (number of FLSs). Otherwise, the lack of cores may cause incorrect results.
Normally, when the accessible resource is limited, one core for at most 2 points will also work for point clouds 
with few hundreds of points.

1. Log into CloudLab
2. At the top left select Experiments
3. Click Start Experiment
4. Select a Profile. Use the profile provided by profile.py
5. Next page: Define the hardware spec of nodes to use. Look to the Manual linked on the page for more information.
6. Define the Number of Nodes desired
7. Next page: You may optionally name the experiment.
8. On the final page: Define the start time and duration of the experiment. If you want to immediately begin the
   experiment do not define a start date or time.


## Software Configurations
Fork our repository and use the forked version.

Go to `cloudlab_vars.sh` and edit the variables accordingly:

- `N`: Number of total nodes in your cluster.
- `HOSTNAME`: The hostname can be found by going to the manifest tab on Cloud Lab and copying the hostname defined 
  there. Copy everything after node-0. For example swarical.nova-PG0.utah.cloudlab.us
- `USERNAME`: CloudLab username
- `REPO_PATH`: The name of your cloned repository, the default value is fine.

Edit the `GITHUB_TOKEN` and `GITHUB_REPO` variables in `scripts/decentralized.sh`. See [GitHub Personal Token 
Tutorial](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) to 
update the GITHUB_TOKEN.

## Setup Nodes for the first time:
Run `scripts/decentralized.sh --copy-key`. This copies your ssh key to the primary node so it can also ssh to 
secondary nodes.

Run `scripts/decentralized.sh --setup`. This should only be run the first time the nodes are initialized. If it says that
you do not have permission to execute this then run chmod +x decentralized_gen.sh.

Navigate to `constants.py` and set PLATFORM to aws.
Navigate to gen_conf.py and edit the RESULTS_PATH to directory shared between your nodes, e.g., 
/proj/nova-PG0/hamedamz/results. This is an example directory, yours will be different depending on your naming scheme.
Change the DURATION variable to the desired experiment duration. This gen_conf.py file uses a template to generate one
configuration file for every combination of values in the list props. With the default code it will generate
configurations to reproduce raw results of Figures 13, 14, and 15 which requires thousands of cores. 

When editing the code, commit and push your changes to your repo and then run `scripts/decentralized_gen.sh 
--update` to 
update all nodes.


SSH into the primary node (node with index 0) and run bash nohup_run.sh. 

Run tail -f my.log to trace the output in the root directory.

To kill the running processes run `scrip/decentralized_gen.sh -- kill`. Then run bash `nohup.kill.sh` on the primary 
node to stop the batch experiments.

The results of the experiment should be written in /proj/nova-PG0/hamedamz/results. This is just an example directory, 
yours
will be slightly different depending on your naming scheme.
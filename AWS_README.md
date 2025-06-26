# Running on Multiple Servers: Amazon AWS

First, set up a cluster of servers. Ideally, the total number of cores of the servers should be equal or greater than
the number of points in the point cloud (number of FLSs).

Set up a multicast domain (For more information on how to create a multicast domain, see aws
docs: https://docs.aws.amazon.com/vpc/latest/tgw/manage-domain.html)

Add your instances to the multicast domain. Use the value of MULTICAST_GROUP_ADDRESS in the constants.py for the group
address.

Ensure you allow all UDP, TCP, and IGMP(2) traffic in your security group.

After setting up AWS:

Choose one of the instances as the primary instance.

Set the private IP address of the primary instance as the `SERVER_ADDRESS` in `constants.py`. Set the `PLATFORM` to 'aws'.

In `aws_vars.sh`, set `N` to the number of total instances you have. Set the `KEY_PATH` as the path to the AWS key pair
on your machine. List the private IP addresses of all the instances in `HOSTNAMES`; the primary should be the first.

In `aws_local_vars.sh`, set `N` to the number of total instances you have. Set the `LOCAL_KEY_PATH` as the path to the
AWS key pair on the primary instance. List the public IP addresses of all the instances in `HOSTNAMES`; the primary
should be the first.

Configure the experiment(s) you want to run by modifying `gen_conf.py`. The current configuration generates
configurations to reproduce raw results of Figures 13, 14, and 15.

Clone the repository and set up the project by running `setup.sh` on each server using the following. Then copy the AWS
key to the primary instance.

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

Finally, use the `utils/file.py` to post-process the results to generate charts:

```
python utils/file.py -i [path to the results directory]
```
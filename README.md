# Swarical

Swarical uses the hardware specification of sensors mounted on FLSs to convert mesh files into point clouds that enable
a swarm of FLSs to localize at the highest accuracy afforded by their hardware.

Authors:  Hamed Alimohammadzadeh(halimoha@usc.edu) and Shahram Ghandeharizadeh (shahram@usc.edu)

## Features

* Three decentralized algorithms that localize FLSs to illuminate a 3D or 2D point cloud.
* A state machine that implements a decentralized algorithm.
* A planner that creates Swarm-Tree and FLS-Trees using K-Means and MST.
* Launches multiple processes, one process per Flying Light Speck, FLS. With large point clouds (FLSs), the software
  scales to utilize multiple servers. Both CloudLab and Amazon AWS are supported.

## Clone

```
git clone https://github.com/flyinglightspeck/Swarical.git
```

## Setup and Install Using Docker (Recommended)

To get started, install Docker Desktop or Docker Engine depending on your operating system:

MacOS: https://docs.docker.com/desktop/setup/install/mac-install/

Ubuntu: https://docs.docker.com/desktop/setup/install/linux/ubuntu/

Once Docker is installed, ensure that it is running before proceeding.

Open a terminal and run the following commands:

```commandline
cd Swarical
bash init.sh
``` 

This script will:

1. Build the Docker image required for the project.
2. Start a Docker container with the necessary environment set up.

Once the script completes, you’ll see a message like the following:

![docker.png](figures%2Fdocker.png)

Copy the provided URL and open it in your web browser. This will take you to a Jupyter Notebook interface running inside
the Docker container.

To run the container after the first time use the following command:
```
docker run -p 8888:8888 swarical
```

## Reproduction of Results

In the Jupyter interface:

1. Open the notebook named reproduction.ipynb.
2. Follow the instructions in each cell step by step.

✅ The notebook will guide you through the process of reproducing the results from the Swarical project.

### Downloading and Viewing Results

After running the experiments using the notebook:

1. In the Jupyter interface (in your browser), navigate to the directory where the output results are saved (given
   by the output logs).
2. Locate the desired file(s) in the file browser.
3. Click the file to select it, then use the Download button located at the top of the file browser to save it to your
   local machine.

![jupyter.png](figures%2Fjupyter.png)

### Running on Multiple Servers

As mentioned in the notebook, running large-scale experiments requires a cluster of machines. We used a cluster of 20
Amazon AWS servers, c6a.metal, with 192 virtual cores.

Please follow the instructions bellow based on your preference:

- Amazon AWS: [AWS_README.md](AWS_README.md)
- CloudLab: [CloudLab_README.md](CloudLab_README.md)

## Setup and Install Using Virtual Environment (Not Recommended)

Note: Using the docker is strongly preferred. To proceed with Virtual Environment make sure you have Python 3.10
installed.

```commandline
cd Swarical
python3.10 -m venv env
```

Then, activate the virtual environment.

```commandline
source env/bin/activate
```

Install the requirements:

```commandline
pip install -r planner-requirements.txt
```

## Mesh Files and Point Clouds

We use mesh files from PRINCETON SHAPE BENCHMARK, e.g., a Chess piece; see `assets/dataset/mesh`. Swarical Planner
takes these mesh files as input and generates point clouds based on its settings.
We provide these point clouds in `assets/dataset/point_cloud`. The value of variable SHAPE in
config.py controls the used point cloud in the Swarical online component. Set the `SHAPE` value to the shape name
(use the file name of .txt files in the `assets` directory as the value of the `SHAPE`,
e.g., `dragon_1147_50_spanning_2_sb`). The repository comes with
the following shapes: `chess`, `dragon`, `kangaroo`, `racecar`, `skateboard`, `grid4x4`.

The file name parts separated by '_' specifies the shape name, number of points, group size, and the planner variant.

## Error with Large Point Clouds

With large point clouds and the Linux operating system, the execution of the software may exhaust the max open files
supported by the operating system.
The provided scripts increase max open files system-wide to be able to run a large
point cloud:

``sudo vim /etc/sysctl.conf``

Add the following line:

``fs.file-max = 9999``

``sudo sysctl -p``

Reload terminal and then run this command:

``ulimit -n 9999``

## Online Localization: ISR, HC, and RSF

ISR, HC, and RSF are three online localization techniques of Swarical. The main difference between the techniques is the
amount of concurrent movements by the FLSs. ISR is superior to HC and RSF. It is faster and more accurate than the
other, minimizing the total distance traveled by FLSs. The use of RSF is not recommended as it fails to localize large
point clouds effectively. The following video demonstrations show each technique localizing the Skateboard. Note that
RSF is not able to fully localize the Skateboard.

* [Inter-Swarm Rounds, ISR](https://youtu.be/GncnoqqYT_w)
* [Highly Concurrent, HC](https://youtu.be/0_Gs7IkDADw)
* [Rounds across the Swarm-tree and FLS-trees, RSF](https://youtu.be/YlLCxW32tvg)

## Position Estimation Using Camera and ArUco Markers
In this implementation, an FLS uses a Raspberry Pi Camera Module 3 NoIR3 with
autofocus and ArUco markers to compute its position relative to another FLS. 
Refer to [this repository](https://github.com/flyinglightspeck/aruco-pose-estimation) for software, instructions, and 3d printable parts required for these experiments.

## Citations

[1] Hamed Alimohammadzadeh, and Shahram Ghandeharizadeh. 2024. Swarical: An Integrated Hierarchical Approach to Localizing
Flying Light Specks. In Proceedings of the 32nd ACM International Conference on Multimedia (MM '24). Association for
Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3664647.3681080

BibTex:

```
@inproceedings{swarical2024, 
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

[2] Hamed Alimohammadzadeh, Shahram Ghandeharizadeh, Federico Cunico,
and Joshua Springer. 2025. Reproducibility Companion Paper: Swarical:
An Integrated Hierarchical Approach to Localizing Flying Light Specks. In
Proceedings of the 33rd ACM International Conference on Multimedia (MM
’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 5 pages.
https://doi.org/10.1145/3746027.3759199

```
@inproceedings{swaricalrepo2025,
author = {Alimohammadzadeh, Hamed and Ghandeharizadeh, Shahram and Cunico, Federico and Springer, Joshua},
title = {{Reproducibility Companion Paper: Swarical: An Integrated Hierarchical Approach to Localizing Flying Light Specks}},
year = {2025},  
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3759199},
doi = {10.1145/3746027.3759199},
booktitle = {Proceedings of the 33rd ACM International Conference on
Multimedia}, 
numpages = {5}, 
location = {Dublin, Ireland}, 
series = {MM '25} 
}
```


## Acknowledgments

This research is supported in part by NSF grants IIS-2232382 and CMMI-2425754. We gratefully acknowledge CloudBank and
CloudLab for the use of their resources.

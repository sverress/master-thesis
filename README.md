# A Reinforcement Learning Approach to Rebalancing and Battery Swaps in Electric Scooter Sharing Systems
This is a repository for the master thesis of Sverre Spetalen, Tord Haflan and Jonas Haga optimizing the operations for
e-scooters with swappable batteries. [Link to github repository](https://github.com/sverress/master-thesis)

## Preface
This thesis concludes our Master of Science at the department of Industrial Economics and Technology Management at the
Norwegian University of Science and Technology. It is written during the spring semester of 2021, and is a continuation
of our specialization project within TIØ4500 Managerial Economics and Operations Research in the fall of 2020. 

We  would  like  to  thank  our  supervisor  Professor  Henrik  Andersson  for  his  valuable  guidance and constructive
feedback throughout the project.  In addition, we would like to express our sincere gratitude towards the industry
operators that have provided us with helpful insight into the industry. A special thanks also goes to 
Professor Keith L. Downing for offering his knowledge on reinforcement learning methods and its applications. 

## Setup
This project uses a virtual environment as python interpreter. To create a virtual environment with the required
packages you need to change directory into the project directory and run to create a new environment named "env" :
```
python -m venv env
```
activate the environment by running (Mac OS):
```
source activate env/bin/activate
```
then run the following command to install all necessary packages
```
pip install -r requirements.txt
```
## Testing
To run all tests run the following command in the root directory of the project
```
python -m unittest discover
```
To run only test for a specific module, run:
```
python -m unittest discover <module_name>
```
e.g.:
```
python -m unittest discover system_simulation
```
## Solstorm Setup and Run
To log in, run:
```
ssh solstorm-login.iot.ntnu.no -l [user name]
```
Start a screen session by running (from login-0-0 node):
```
screen -S <screen_name>
```
Or reattach to a previous screen: 
```
screen -r <screen_ID>
```
To list availible screens, run: 
```
screen -ls
```
From an active screen, press "Ctrl + a + d" to detach from the screen.

IMPORTANT: You must be at a compute node to run files. Nodes 4-50 to 4-59 and all nodes on rack 6 are not allowed for student use. To go to a compute node while in a screen, run (goes to node 1-2): 
```
ssh compute-1-2
```

Go to the master-thesis folder:
```
cd /storage/users/<username>/master-thesis
``` 
Run regular git commands to pull, and python commands to run the desired files.

To scroll: "ctrl + A" (activate input) then "esc". Then you can scroll. Press "esc" again to escape scroll mode.
 
## How to run the code

All scripts used to train and evaluate the system are located in the analysis module

To train models, modify kwargs arguments for the world object in train_value_function.py and run the file. For
multiprocessing, use the multiprocessing_training.py. 

To evaluate models run the following command:
```
python analysis/evaluate_policies.py <world attribute> <path to trained models directory>
```
The world attribute dictates how the end evaluation graph should label the different models. The path to trained models
is the path to the folder on your computer where the trained models are saved. See evaluate_policies.py for details and
further configuration of the run
# Master Thesis
This is a repository for the master thesis of Sverre Spetalen, Tord Haflan and Jonas Haga optimizing the operations for
e-scooters with swappable batteries.
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
 


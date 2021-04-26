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

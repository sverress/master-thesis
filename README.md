# Master Thesis
This is a repository for the master thesis of Sverre Spetalen, Tord Haflan and Jonas Haga optimizing the operations for
e-scooters with swappable batteries.
##Setup

### Gurobi
This project uses gurobi as a solver for the optimization problem. To use gurobi we need a license file and the program
itself. [This quickstart](https://www.gurobi.com/wp-content/plugins/hd_documentations/content/pdf/quickstart_mac_8.1.pdf#page=89&zoom=100,96,96)
 is a good starting point to obtain both of these.

### Anaconda environment
It is recommended to use anaconda for gurobi. [Download anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) 
(I used the command-line install) and create an environment from a environment.yml file.
Change directory into the root directory and run the following command:
```
conda env create -f environment.yml
```
This command creates a conda environment named "master" with all the required packages for the project.
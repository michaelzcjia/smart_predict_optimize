# smart_predict_optimize

smart_predict_optimize is a Python implementation of the "Smart, Predict then Optimize" (Elmachtoub and Grigas) framework that aims to the compare the computational efficiency of optimizing with a Linear Program versus optimizing with Stochastic Gradient Descent.

## Package installation

All packages that are used can be found in requirements.txt. To install all packages, run the following:
```bash
pip install requirements.txt
```

## Code Overview

### HelperFunctions.py
Contains the majority of the code. Contains functions and classses to:
- Generate synthetic data and formulate it into a Shortest Path problem.
- Compute SPO and SPO+ loss.
- Solve a Shortest Problem problem as a Linear Program
- Solve a Shortest Path problem using Stochastic Gradient Descent

### Experiments.py
Generates data and runs experiments using function and classes from HelperFunction.py. Generates a .pkl file (SPOresultsAllVars.pkl) of the experiment output. 

### ResultsAnalysis.ipynb 
Visualizes the results of the experiments generated from Experiments.py. Reads SPOresultsAllVars.pkl and writes output plots to the "plots" directory. 

## Usage

To reproduce the plots, follow the steps below: 
1. Generate experiment data. Note that running this code will take a significant amount of time (24+ hours) on standard consumer hardware. 

```bash
python HelperFunctions.py 
```
2. Generate plots by running each cell in ResultsAnalysis.ipynb. 

## Project Status 

This project is completed and not actively being worked on. 

# smart_predict_optimize

smart_predict_optimize is a Python implementation of the "Smart, Predict then Optimize" (Elmachtoub and Grigas, 2021) framework that aims to compare the computational efficiency and performance of optimizing a linear SPO model with a Linear Program versus Stochastic Gradient Descent.

## Package installation

All packages that are used can be found in requirements.txt. To install all packages, run the following:
```bash
pip install requirements.txt
```

## Code Overview

### HelperFunctions.py
Contains the majority of the code. Contains functions and classes to:
- Generate synthetic data and formulate it into a Shortest Path problem.
- Compute SPO and SPO+ loss.
- Solve a shortest path problem
- Fit a linear model to predict the parameters of a shortest path problem with the SPO+ loss function via a Linear Program
- Fit a linear model to predict the parameters of a shortest path problem with the SPO+ loss function via Gradient Descent

### Experiments.py
Generates data and runs experiments using functions and classes from HelperFunction.py. Generates a .pkl file (SPOresultsAllVars.pkl) of the experiment output. 

### ResultsAnalysis.ipynb 
Visualizes the results of the experiments generated from Experiments.py. Reads SPOresultsAllVars.pkl and writes output plots to the "plots" directory. 

## Usage

To reproduce the plots, follow the steps below: 
1. Generate experiment data. Note that running this code will take a significant amount of time (20+ hours) on standard consumer hardware. 

```bash
python Experiments.py 
```
2. Generate plots by running each cell in ResultsAnalysis.ipynb. 

Alternatively plots can be recreated by using the SPOresultsAllVars.pkl file that is in this repository.

## Project Status 

This project is completed and not actively being worked on. 

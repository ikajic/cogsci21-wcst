# Spiking neural WCST model

This repository contains the source code for the spiking neural model of the WCST implemented in Nengo, and published in Kajić, I. and Stewart, T.C (2021): Biologically Constrained Large-Scale Model of the Wisconsin Card Sorting Test (Proceedings of the 43th Annual Meeting of the Cognitive Science Society).

## Getting started

Start by cloning this repository and installing the Python packages in `requirements.txt`:

```
$ pip install -r requirements.txt
```

From the cloned repository, import the model and run it: 

```
from model import WCSTModel

result = WCSTModel().run(T=5, d=64)
```

It will also create a model with 64-dimensional semantic pointers, and run the simulation for 5 seconds. 
Under the hood, this also creates an experimental module that controls the presentation of cards to the model. To see default parameters (such as how many cards there are in one category, or how many cards there are in a deck 
The result of the simulation will be stored as pandas DataFrame `result` as well as printed to the screen. This shouldn't take more than 5 minutes, even on less powerful machines.

The resulting output should look like this:



## Getting serious


Ideally, the model should be run with the `nengo_ocl` backend to take advantage of GPU optimizations that speed up simulation run times. Results presented in the paper were generated with simulations ran in such a way. The script `run-all.py` shows an example of configuration for running large-scale simulations. [nengo_ocl](https://github.com/nengo-labs/nengo-ocl) repository has installation instructions for those wishing to experiment with this setup.

As well, to run the `run-all` script one needs to manually check out [ocl-use-context](https://github.com/ctn-waterloo/ctn_benchmarks/tree/ocl-use-context) branch in the ctn_benchmarks repository, which allows `nengo_ocl` to be provided as the backend.

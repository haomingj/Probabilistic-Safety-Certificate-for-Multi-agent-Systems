# Probabilistic Safety Certificate for Multi agent Systems
This repository stores (an improved version) of the simulation code for the conference paper [Probabilistic Safety Certificate for Multi-agent Systems](https://ieeexplore.ieee.org/abstract/document/9992692).
## Requirements
The code is tested with CUDA 11.8, ```python=3.7.16```, ```pycuda=2021.1```, and ```numpy=1.19.2```. It should also work with other versions of these packages.
## Running the code
To start the simulation, put all code in the same directory and run 
```bash
python main.py save_directory
```
```save_directory``` is the directory where the simulation data is stored, and has to be already created at the time the simulation starts.

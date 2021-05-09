import nengo_ocl
import os
import datetime

from pathlib import Path
from model import WCSTModel
from xsetup import Feedback

import pyopencl as cl

results_dir = 'results/'

# Experimental paramaters
seq_correct = 10
timesteps = 300
deck_size = 128
random_rule = True

# Simulation parameters
nr_simulations = 200
sim_len = 100
gpu_to_use = 3 # which gpu to use for this sim, not how many

# Model parameters
dimensions = 512
feedback_rule_strength=0.7
feedback_gate_strength=0.9

dirname = '{}d-{}frs-{}fgs'.format(
    dimensions, feedback_rule_strength, feedback_gate_strength)

platform = cl.get_platforms()
gpu = platform[0].get_devices(device_type=cl.device_type.GPU)[gpu_to_use]
ctx = cl.Context(devices=[gpu])

sim_dir = os.path.join(os.getcwd(), results_dir, dirname)
Path(sim_dir).mkdir(parents=True, exist_ok=True)
print('Created {}'.format(sim_dir))

for seed in range(nr_simulations):
    print('Running simulation #{}'.format(seed))  
    
    WCSTModel().run(
        x_seq_correct=seq_correct,
        x_timesteps=timesteps,
        x_deck_size=deck_size,
        x_random_rule=random_rule,
        feedback_rule_strength=feedback_rule_strength,
        feedback_gate_strength=feedback_gate_strength,
        mseed=seed,
        d=dimensions,
        T=sim_len,        
        data_dir=sim_dir,
        backend='nengo_ocl',
        context=ctx)   # context only works for ocl-use-context branch (https://github.com/ctn-waterloo/ctn_benchmarks/tree/ocl-use-context)

import json
import os
import numpy as  np

test_dirs = ["debug_results/test-func_rastrigin-dim_1-kern_Matern32-runs_2-iters_51-init_samp_50-noise_0.100000-exp_bias_0.250000_01-23-22_23:45:53",
             "debug_results/skopt-func_rastrigin-dim_1-kern_Matern32-runs_1-iters_51-init_samp_50-exp_bias_0.0_01-23-22_23:49:46"]

configs = []
results = []
for d in test_dirs:
    with open(os.path.join(d, "config.json")) as f:
        configs.append(json.load(f))
    with open(os.path.join(d, "log.json")) as f:
        results.append(json.load(f))
    
for i, r in enumerate(results):
    print(configs[i]["system"])
    

import pdb; pdb.set_trace()


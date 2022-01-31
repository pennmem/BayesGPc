import json
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

test_dirs = ["debug_results/test-func_rastrigin-dim_1-kern_Matern32-runs_2-iters_51-init_samp_50-noise_0.100000-exp_bias_0.250000_01-23-22_23:45:53",
             "debug_results/skopt-func_rastrigin-dim_1-kern_Matern32-runs_1-iters_51-init_samp_50-exp_bias_0.0_01-23-22_23:49:46"]

# tips = sns.load_dataset("tips")

# ax = sns.boxplot(x=tips["total_bill"])

# ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                  data=tips, palette="Set3")

# import pdb; pdb.set_trace()

# plt.show()

# exit()

configs = []
results = []
run_results = []
run_df = pd.DataFrame()

for i, d in enumerate(test_dirs):
    with open(os.path.join(d, "config.json")) as f:
        configs.append(json.load(f))
    with open(os.path.join(d, "log.json")) as f:
        results.append(json.load(f))
    
    n_runs = configs[-1]["n_runs"]
    res = results[-1]
    # import pdb; pdb.set_trace()
    model = configs[-1]["system"]  #configs[-1]["tag"]
    excluded_func = []
    for f, fv in res.items():
        if f in excluded_func: continue
        temp = dict()
        temp["Relative Error"] = np.array(fv["relative errors"])[:, -1]
        temp["Max Sample Run Times"] = np.array(fv["sample times"])[:, -5:].mean(axis=1)
        temp["Function"] = [f] * n_runs
        temp["Model"] = [model] * n_runs
        run_df = run_df.append(pd.DataFrame(temp))
    
# for i, r in enumerate(results):
#     print(configs[i]["system"])

# sns.violinplot(x="Function",
#             y="Relative Error",
#             hue="Model",
#             data=run_df
#             )
sns.boxplot(x="Function",
            y="Relative Error",
            hue="Model",
            data=run_df,
            notch=True, 
            bootstrap=2000
            )
plt.title("Relative Errors")

# plt.figure()
# sns.boxplot(x="Function",
#             y="Max Sample Run Times",
#             hue="Model",
#             data=run_df
#             )

plt.show()

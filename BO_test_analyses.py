import json
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# test_dirs = ["debug_results/test_CBayes-func_schwefel-dim_1-kern_Matern32-runs_5-iters_150-init_samp_50-noise_0.100000-exp_bias_0.250000_03-18-22_12:12:41",
#              "debug_results/test_PyRef-impl_skopt-func_schwefel-dim_1-kern_Matern32-runs_5-iters_150-init_samp_50-noise_0.1-exp_bias_0.25_03-18-22_12:25:06"]
test_dirs = None

if test_dirs is None:
    cluster_run_dirs = ["results/",
                        "results/"]

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
    model = configs[-1]["impl"]  #configs[-1]["tag"]
    excluded_func = []
    for f, fv in res.items():
        if f in excluded_func: continue
        temp = dict()
        temp["Relative Error"] = np.array(fv["relative errors"])[:, -1]
        temp["Max Sample Run Times"] = np.array(fv["sample times"])[:, -5:].mean(axis=1)
        temp["Function"] = [f] * n_runs
        temp["Model"] = [model] * n_runs
        # other config parameters for comparison
        for k in ["exp_bias", "n_iters", "n_init_samp", "n_init_samples", "noise_level", "kern", "kernel"]:
            if k not in configs[-1].keys():
                continue
            if k == "n_init_samples":  # handle special case in naming differences
                temp["n_init_samp"] = [configs[-1][k]] * n_runs
            elif k == "kernel":  # handle special case in naming differences
                temp["kern"] = [configs[-1][k]] * n_runs
            else:
                temp[k] = [configs[-1][k]] * n_runs
        run_df = run_df.append(pd.DataFrame(temp))
    
# for i, r in enumerate(results):
#     print(configs[i]["system"])

print(run_df.columns)

# sns.violinplot(x="Function",
#             y="Relative Error",
#             hue="Model",
#             data=run_df
#             )

# sns.boxplot(x="Function",
#             y="Relative Error",
#             hue="Model",
#             data=run_df,
#             # notch=True,
#             # bootstrap=2000
#             )
# plt.title("Relative Errors")

# print(run_df.loc[:, ["Model", "kern"]])

for eb in run_df["exp_bias"].unique():
    for nl in run_df["noise_level"].unique():
        for k in run_df["kern"].unique():
            for iters in run_df["n_iters"].unique():
                for init in run_df["n_init_samp"].unique():
                    plt.figure()
                    sns.boxplot(x="Function",
                                y="Relative Error",
                                hue="Model",
                                data=run_df,
                                # notch=True,
                                # bootstrap=2000
                                )
                    title = f"Relative Errors for exp_bias: {eb}, noise: {nl},\nkern: {k}, samples: {iters}, init_samples: {init}"
                    print(title)
                    plt.title(title)

# plt.figure()
# sns.boxplot(x="Function",
#             y="Max Sample Run Times",
#             hue="Model",
#             data=run_df
#             )

plt.show()

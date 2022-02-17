import os

import numpy as np
from numpy.random.mtrand import sample
import pandas as pd
import pingouin as pg

if __name__ == "__main__":
    test = "welch_anova"
    output_dict = dict()
    output_dict["n_tests"] = np.array([2.0])
    output_dict["n_groups"] = np.array([2.0])
    data = dict()
    sample_sizes = [10, 8]
    mean_diff = 3
    group_data = [np.arange(sample_sizes[0]) - sample_sizes[0]/2 - 0.5,
                  np.arange(sample_sizes[1]) - sample_sizes[1]/2 - 0.5 + mean_diff]
    data["x"] = []
    for i in range(int(output_dict["n_groups"])):
        data["x"].extend([i for _ in range(sample_sizes[i])])
        output_dict[f"x{i}_0"] = group_data[i]
    data["y"] = np.concatenate(group_data, axis=0)

    data = pd.DataFrame(data)
    # print(data)

    aov_res = pg.welch_anova(data, dv="y", between="x")
    # print(aov_res)
    # print("F: ", aov_res["F"][0])
    # print("p-val: ", aov_res["p-unc"][0])

    output_dict["stat0"] = np.array([aov_res["F"][0]])
    output_dict["p0"] = np.array([aov_res["p-unc"][0]])
    output_dict["dof0"] = np.array([aov_res["ddof2"][0]])

    data = dict()
    sample_sizes = [15, 12]
    mean_diff = -3
    group_data = [np.arange(sample_sizes[0]) - sample_sizes[0]/2 - 0.5,
                  np.arange(sample_sizes[1]) - sample_sizes[1]/2 - 0.5 + mean_diff]
    data["x"] = []
    for i in range(int(output_dict["n_groups"])):
        data["x"].extend([i for _ in range(sample_sizes[i])])
        output_dict[f"x{i}_1"] = group_data[i]
    data["y"] = np.concatenate(group_data, axis=0)

    data = pd.DataFrame(data)
    # print(data)

    aov_res = pg.welch_anova(data, dv="y", between="x")
    # print(aov_res)
    # print("F: ", aov_res["F"][0])
    # print("p-val: ", aov_res["p-unc"][0])

    output_dict["stat1"] = np.array([aov_res["F"][0]])
    output_dict["p1"] = np.array([aov_res["p-unc"][0]])
    output_dict["dof1"] = np.array([aov_res["ddof2"][0]])

    fname = "testCSearchComparison_" + test + ".npz"
    path = "np_files"
    np.savez(os.path.join(path, fname), **output_dict)
    print("Saved", fname)


    test = "welch_ttest"
    output_dict = dict()
    output_dict["n_tests"] = np.array([2.0])
    output_dict["n_groups"] = np.array([2.0])
    sample_sizes = [10, 8]
    mean_diff = 3
    x = np.arange(sample_sizes[0]) - sample_sizes[0]/2 - 0.5
    y = np.arange(sample_sizes[1]) - sample_sizes[1]/2 - 0.5 + mean_diff

    ttest_res = pg.ttest(x, y, 
                         paired=False, alternative="greater", correction=True)
    # print(ttest_res)
    # print("T: ", ttest_res["T"][0])
    # print("p-val: ", ttest_res["p-val"][0])

    output_dict["x0_0"] = x
    output_dict["x1_0"] = y
    output_dict["stat0"] = np.array([ttest_res["T"][0]])
    output_dict["p0"] = np.array([ttest_res["p-val"][0]])
    output_dict["dof0"] = np.array([ttest_res["dof"][0]])

    sample_sizes = [15, 12]
    mean_diff = -3
    x1 = np.arange(sample_sizes[0]) - sample_sizes[0]/2 - 0.5
    y1 = np.arange(sample_sizes[1]) - sample_sizes[1]/2 - 0.5 + mean_diff

    ttest_res = pg.ttest(x1, y1, 
                         paired=False, alternative="greater", correction=True)
    # print(ttest_res)
    # print("T: ", ttest_res["T"][0])
    # print("p-val: ", ttest_res["p-val"][0])

    output_dict["x0_1"] = x1
    output_dict["x1_1"] = y1
    output_dict["stat1"] = np.array([ttest_res["T"][0]])
    output_dict["p1"] = np.array([ttest_res["p-val"][0]])
    output_dict["dof1"] = np.array([ttest_res["dof"][0]])

    fname = "testCSearchComparison_" + test + ".npz"
    path = "np_files"
    np.savez(os.path.join(path, fname), **output_dict)
    print("Saved", fname)


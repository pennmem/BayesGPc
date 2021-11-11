# playing around with Gaussian processes

import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
import sklearn.gaussian_process.kernels as kernels
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from functools import partial

def dict_product(di, list_keys=None, prod_di=dict(), prod_list=list(), key_idx=0):
    """
    Implements itertools.product for dictionaries of lists
    Param di: dict[key]->list
    returns: list of dictionaries mapping from keys to respective elements in a Cartesian product. 
    """
    if key_idx == 0:
        list_keys = list(di.keys())
    
    if key_idx == len(list_keys):
        prod_list.append(prod_di)
    else:
        key = list_keys[key_idx]
        for el in di[key]:
            dict_product(di, list_keys, {**prod_di, key:el}, prod_list, key_idx+1)

    if key_idx == 0:
        return prod_list


kerns = dict()
kerns["matern32"] = {"kern": partial(kernels.Matern, nu=1.5),
                   "hps": {"length_scale": [0.01, 0.5, 1.0, 2.0, 100.0]}}

# test data sets
range_interval = (-1, 1)
# def test_func(x):
#     return (1-x**2).reshape(-1)

# for varying degrees of variation or smoothness over function domain
def test_func(x):
    return (np.sin(x**2)).reshape(-1)

noise_scale = 0.1
def observation_noise(shape):
    return noise_scale * (range_interval[1] - range_interval[0]) * np.random.randn(*shape)

x_interval = np.array([0, 4])
num_samples = 100

# X_train = np.random.rand(num_samples, 1) * np.diff(x_interval) + x_interval[0]
X_train = np.linspace(x_interval[0], x_interval[1], num_samples).reshape(-1, 1)
y_train = test_func(X_train) + observation_noise((num_samples,))

subplot_cols = 4

for kern_name, kern_dict in kerns.items():
    # chosen kernel hyperparameters

    pars_list = dict_product(kern_dict["hps"])
    subplot_rows = (len(pars_list)+1)//subplot_cols + 1
    subplot = 1
    plt.figure(figsize=(20, subplot_rows*5))
    for pars in pars_list:
        pars_w_bounds = {**pars, **{(k + "_bounds"): "fixed" for k, v in pars.items()}}
        kern = kern_dict["kern"](**pars_w_bounds) + kernels.WhiteKernel()

        gp = GPR(kern)
        gp.fit(X_train, y_train)

        x_plot = np.expand_dims(np.linspace(x_interval[0], x_interval[1], 500), -1)
        y_plot, std_plot = gp.predict(x_plot, return_std=True)
        plt.subplot(subplot_rows, subplot_cols, subplot)
        subplot += 1
        plt.plot(x_plot, y_plot, label="mu: " + str(list(pars.values())), color="b")
        plt.fill_between(x_plot.reshape(-1), 
                         y_plot.reshape(-1)+std_plot, 
                         y_plot.reshape(-1)-std_plot, 
                         alpha=0.3, color='r')
    
        plt.scatter(X_train, y_train, marker='x', label="Samples", c='k')
        plt.plot(x_plot, test_func(x_plot), '--', label="True objective", c='g')
        plt.legend()
        plt.title("GPR for " + kern_name + "\n" + \
            "Param order: " + str(list(pars.keys()))+ \
            "  logL: {:.4}".format(gp.log_marginal_likelihood_value_))
        plt.xlabel("x")

    # fit hyperparameters
    kern = kern_dict["kern"]() + kernels.WhiteKernel()
    gp = GPR(kern)
    gp.fit(X_train, y_train)
    pars = {k: v for k, v in gp.kernel_.get_params().items() if k in list(pars.keys())}

    x_plot = np.expand_dims(np.linspace(x_interval[0], x_interval[1], 500), -1)
    y_plot, std_plot = gp.predict(x_plot, return_std=True)
    plt.subplot(subplot_rows, subplot_cols, subplot)
    subplot += 1
    plt.plot(x_plot, y_plot, label="mu: " + str(list(pars.values())), color="b")
    plt.fill_between(x_plot.reshape(-1), 
                        y_plot.reshape(-1)+std_plot, 
                        y_plot.reshape(-1)-std_plot, 
                        alpha=0.3, color='r')

    plt.scatter(X_train, y_train, marker='x', label="Samples", c='k')
    plt.plot(x_plot, test_func(x_plot), '--', label="True objective", c='g')
    plt.legend()
    plt.title("GPR for " + kern_name + "\n" + \
        "Param order: " + str(list(pars.keys()))+ \
        "  logL: {:.4}".format(gp.log_marginal_likelihood_value_))
    plt.xlabel("x")

plt.show()

'''
TODO
look at effects of hyperparameter choices for main kernels
add fit models
look at effects of alpha and white kernels
look at choices of initial values for HP fitting, might be problematic
'''

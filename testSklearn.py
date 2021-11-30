import numpy as np
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt

import os

# generate or load data
# number of features
m = 1
# number of samples
n = 100

y_noise_scale = 0.3
X_scale = 25

RANDOM_STATE = 122

np.random.seed(RANDOM_STATE+1)
X = np.random.rand(n, m) * X_scale - 0.5*X_scale
X2 = np.random.rand(n//2, m) * X_scale - 0.5*X_scale

if m == 1:
    def fun(X):
        return np.sin(X)/X
else:
    raise NotImplementedError

y = fun(X)
y += np.random.randn(*y.shape) * np.std(y) * y_noise_scale
y_mean = y.mean()
y_std = y.std()
y -= y_mean
y /= y_std

# build model
sigma_0 = 1.0
noise_level = 1.0

# generate test of either model with arbitrary hyperparameters or trained model
train = False

if train:
    # for generating tests of GPR, uncomment white kernel and one of the non-trivial kernels
    kern = gp.kernels.WhiteKernel(noise_level=noise_level)

    # kern += gp.kernels.Matern() * gp.kernels.ConstantKernel(); kern_name = "matern32"
    # param_permutation = np.arange(3)

    # kern += gp.kernels.Matern(nu=2.5) * gp.kernels.ConstantKernel(); kern_name = "matern52"
    # param_permutation = np.arange(3)

    # kern += gp.kernels.RBF() * gp.kernels.ConstantKernel(); kern_name = "rbf"
    # param_permutation = np.arange(3)

    # kern += gp.kernels.RationalQuadratic(alpha_bounds=(1e-05, 1e20)) * gp.kernels.ConstantKernel(); kern_name = "ratquad"
    # param_permutation = np.arange(3)

    # p = 2
    # kern += gp.kernels.Exponentiation(gp.kernels.DotProduct(sigma_0=sigma_0), p) * gp.kernels.ConstantKernel(); kern_name = "poly"
    # param_permutation = np.arange(3)
else:
    # for generating tests of individual kernels, uncomment one of the kernels.
    # use fixed hyperparameters for all kernels and multiply by constant kernel to implement variance hyperparameter in GCp

    # length_scale=np.random.rand() * 2
    # kern = gp.kernels.Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=1.5); kern_name = "matern32"
    # kern_clone = gp.kernels.Matern(length_scale=length_scale, nu=1.5)

    # length_scale=np.random.rand() * 2
    # kern = gp.kernels.Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5); kern_name = "matern52"
    # kern_clone = gp.kernels.Matern(length_scale=length_scale)

    # kern = gp.kernels.RBF(length_scale=np.random.rand() * 2, length_scale_bounds="fixed"); kern_name = "rbf"

    # kern = gp.kernels.RationalQuadratic(length_scale=np.random.rand() * 2, length_scale_bounds="fixed",
    #                                     alpha=np.random.rand() * 2, alpha_bounds="fixed"); kern_name = "ratquad"

    p = 2
    kern = gp.kernels.Exponentiation(gp.kernels.DotProduct(sigma_0=np.random.rand() * 2, sigma_0_bounds="fixed"), p); kern_name = "poly"

    # make kernel clones with non-fixed hyperparameters so that gradients can be computed
    constant_value = np.random.rand() * 2
    constKern = gp.kernels.ConstantKernel(constant_value=constant_value, constant_value_bounds="fixed")
    # constKern_clone = gp.kernels.ConstantKernel(constant_value=constant_value)
    # multiply by constant kernels to implement variance hyperparameter in GCp
    kern *= constKern
    # kern_clone *= constKern_clone

# print(kern_clone.get_params())
# print(kern.get_params())
obsNoiseVar = np.array([1e-1,])
gpr = gp.GaussianProcessRegressor(kernel=kern, normalize_y=True, random_state=RANDOM_STATE+2, alpha=obsNoiseVar)
# must train GPR to access hyperparameter gradients and log-likelihoods
# (even for testing with arbitrary hyperparameter values)
gpr.fit(X, y)

# print(gpr.kernel_.theta)
KX, gKX = gpr.kernel_(X, eval_gradient=True)
# print(KX[:3, :3])
# KX, gKX = kern_clone(X, eval_gradient=True)

# print(KX[:3, :3])
# print(gKX.shape)

# exit()

K_diag = np.expand_dims(gpr.kernel_.diag(X), 1)
KX1_2 = gpr.kernel_(X, X2)

theta_rand = np.log(np.random.rand(*kern.theta.shape) * 2.0)
import copy
kern_rand = copy.deepcopy(kern)
kern_rand.theta = theta_rand

logL_rand, gradTheta_rand = gpr.log_marginal_likelihood(kern_rand.theta, eval_gradient=True)
# TODO permute gradTheta_rand as needed for particular kernels
logL, gradTheta = gpr.log_marginal_likelihood(gpr.kernel_.theta, eval_gradient=True)
# convert gradients for GCp implementation which uses inverse_width = 1/length_scale^2
if train:
    if kern_name == "rbf":
        gradTheta_rand[1] *= -0.5
        gradTheta[1] *= -0.5

print("Kernel:\n", gpr.kernel_)
print(f"theta log-marginal likelihood: {gpr.log_marginal_likelihood_value_:.3f}")

if m == 1:
    X_pred = np.linspace(-0.8*X_scale, 0.8*X_scale, 1000).reshape(-1, 1)
    y_pred, std_pred = gpr.predict(X_pred, return_std=True)
    y_pred = y_pred.reshape(-1)
    y_true = (fun(X_pred) - y_mean)/y_std
    print(f"Score: {gpr.score(X_pred, y_true):.3f}")

    lw = 2
    plt.scatter(X, y, s=3, c='k', label='data')
    plt.plot(X_pred, y_true, color='navy', lw=lw, label='True')
    plt.plot(X_pred, y_pred, color='darkorange', lw=lw,
            label='GPR (%s)' % gpr.kernel_)
    plt.fill_between(X_pred[:, 0], y_pred - std_pred, y_pred + std_pred, color='darkorange',
                     alpha=0.2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('GPR')
    plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
    plt.show()


if train:
    print()
    print("log-likelihood ", logL)
    print()
    print("parameter gradient ", gradTheta)
    param_permutation = np.array([0,2,1])
    print("permuted parameter gradient (to align with CGp order) ", gradTheta[param_permutation])

    print()
    print("log-likelihood random params ", logL_rand)
    print()
    print("gradient random parameters ", gradTheta_rand)
    param_permutation = np.array([0,2,1])
    print("permuted gradient random parameters (to align with CGp order) ", gradTheta_rand[param_permutation])
    print()

output_dict = {"test:"+("fit" if train else "inference"): np.array([0]), "X": X,"X2": X2, "y": y, "X_pred": X_pred, 
               "y_pred": y_pred, "std_pred": std_pred, "bias": 0, "scale": 1, "numActive": -1, "approxInt": 0, 
               "K": KX, "K1_2": KX1_2, "K_diag": K_diag, "gradX": gKX, "logL": logL, "gradTheta": gradTheta,
               "logL_rand": logL_rand, "gradTheta_rand": gradTheta_rand, "theta_rand": kern_rand.theta,
               "obsNoiseVar": obsNoiseVar}

# TODO this is a mess... rewrite for arbitrary kernels
def add_kernel_attributes(output_dict, kern):
    for k, v in kern.get_params().items():
        # cnpy doesn't automatically handle tuples properly, so skip bounds on 
        # hyperparameters for now (bounds not currently used in testing GCp)
        if "_bounds" in k:
            continue
        if train:
            if isinstance(v, gp.kernels.Kernel):
                if "__" in k: # skip composite kernels
                    continue
                # only composite kernel handled is product of non-trivial kernel with constant (+ white noise kernel)
                if "*" in str(v) and " + " not in str(v):
                    # get keys of parameters associated with product kernel
                    for ki, vi in kern.get_params().items():
                        # print(k, ki)
                        if (not isinstance(vi, gp.kernels.Kernel)) and len(ki) >= len(k) and ki[:len(k)] == k:
                            output_dict[k + "__" + ki.split("__")[-1]] = vi
                print(str(v))
                output_dict[k + "&" + str(v).split("(")[0]] = np.array([0])
            else:
                # will place some parameters in the dictionary redundantly
                output_dict[k] = v
        else:  # deal with sklearn composite kernel separately for two kernels instead of three
            if k[:2] == "k2" and len(k) > 2:
                output_dict["k1" + k[2:]] = v
            elif isinstance(v, gp.kernels.Kernel) and k != "k2":
                output_dict[k + "&" + str(v).split("(")[0]] = np.array([0])
            elif k == "k2":
                pass
            else:
                # will place some parameters in the dictionary redundantly
                output_dict[k] = v

add_kernel_attributes(output_dict, gpr.kernel_)

# print([(k, v) for k, v in output_dict.items() if k not in ["y", "X_pred", "y_pred", "std_pred", "bias", "scale", \
#                                                   "numActive", "approxInt", "X", "X2", "kX", "gradX", "logL", "gradTheta"]])
print([(k, v) for k, v in output_dict.items() if k not in ["y", "X_pred", "y_pred", "std_pred", \
                                                  "X", "X2", "kX", "gradX", "K", "K1_2", "K_diag", "gradX", "logL", "gradTheta"]])
# print({k: v for k, v in output_dict.items() if k not in ["y", "y_pred", "std_pred", "bias", "scale", "numActive", "approxInt", "X"]})

# save data and results
fname = "testSklearn_" + ("gpr_" if train else "kernel_") + kern_name + ".npz"
path = "np_files"
np.savez(os.path.join(path, fname), **output_dict)

# test_load = dict(np.load(os.path.join(path, fname), allow_pickle=True))
# print(test_load)

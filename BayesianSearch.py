import numpy as np
from scipy.optimize import minimize, dual_annealing
from scipy.stats import norm
import matplotlib.pyplot as plt
from functools import partial


class BayesianSearch:
    def __init__(self, gp, acq_func, init_samples=5, num_restarts=1, x_dim=1, bounds=[], tol=1e-9):
        self.gp = gp
        # function of x, y_best, and GP surrogate model (modeling y_test|x_test, y, X)
        self.acq_func = acq_func
        self.neg_acq_func = lambda *args, **kwargs: -self.acq_func(*args, **kwargs)
        self.init_samples = init_samples
        self.num_restarts = num_restarts
        self.samples_x = np.array([[]]).reshape((0, x_dim))
        self.samples_y = np.array([]).reshape((0,))
        self.y_best = -float("inf")
        self.x_dim = x_dim
        self.bounds = bounds
        self.tol = tol
        self.fit = False

    def get_next_sample(self):
        if self.samples_x.shape[0] < self.init_samples:
            # TODO return grid points or something more intelligent for low numbers of initial samples
            # TODO get Latin hypercube/orthogonal sampling for larger numbers of initial samples
            bounds = np.array(self.bounds)
            return np.random.rand(1, self.x_dim) * np.swapaxes(np.diff(bounds, axis=1), 0, 1) + bounds[:, 0]

        # could use rank-1 inverse updates for efficiency, would require significant internal updating in GPc and Python
        # would need to check numerical stability
        self.gp.fit(self.samples_x, self.samples_y)
        self.fit = True

        # optimize acquisition function to obtain next sample, x

        # TODO add multiple random restarts for local optimizer (e.g. minimize()) or use global optimizer
        # x0 = np.zeros((self.x_dim,))
        # res = minimize(self.acq_func, x0, (self.y_best, self.gp,), method="SLSQP", bounds=self.bounds, tol=self.tol)
        res = dual_annealing(self.neg_acq_func, bounds=self.bounds, args=(self.y_best, self.gp,), maxiter=50, seed=1)

        if res.success:
            pass
            # TODO random restarts on failure
        
        return np.array(res.x).reshape((1, self.x_dim))

    def add_sample(self, sample_x, sample_y):
        if self.samples_x.shape[0] == 0:
            self.samples_x = sample_x
            self.samples_y = sample_y
        else:
            self.samples_x = np.concatenate([self.samples_x, sample_x], axis=0)
            self.samples_y = np.concatenate([self.samples_y, sample_y], axis=0)
        if sample_y > self.y_best:
            self.y_best = sample_y
        self.fit = False

    def plot(self):
        # plot the current samples, GP model, acquisition function, and next best sample.
        # currently implemented for only 1D functions
        x_plot = np.expand_dims(np.linspace(self.bounds[0][0], self.bounds[0][1], 200), -1)
        if not self.fit:
            self.gp.fit(self.samples_x, self.samples_y)
        y_plot, std_plot = self.gp.predict(x_plot, return_std=True)
        plt.plot(x_plot, y_plot, label="mu", color='b')
        plt.fill_between(x_plot.reshape(-1),
                         y_plot.reshape(-1)+std_plot,
                         y_plot.reshape(-1)-std_plot,
                         alpha=0.3, color='r')
        
        plt.scatter(self.samples_x, self.samples_y, marker='x', label="Samples", c='k')
        acq_func_vals = self.acq_func(x_plot, self.y_best, self.gp)
        acq_func_vals = acq_func_vals - np.min(acq_func_vals)
        acq_func_vals /= np.max(acq_func_vals)
        acq_func_vals -= range_interval[0] + 2
        plt.plot(x_plot, acq_func_vals, color='g', label="Acquisition function")
        plt.plot(x_plot, std_plot - range_interval[0] - 2, color="b", label="std")
        x_next = self.get_next_sample()

        y_next = self.gp.predict(x_next)
        plt.scatter(x_next, y_next, marker='x', color='r', label="Next sample")
        plt.legend()
        plt.title("State of Bayesian search run")
        plt.xlabel("x")


def ExpectedImprovement(x, y_best, gp, exploration_bias=0):
    """
    Param x: ndarray (n_samples, x_dim) or (x_dim,). Input x value for evaluation.
    Param y_best: float. Current best outcome.
    Param gp: sklearn.gaussian_process.GaussianProcessRegressor. Fit GP model.
    Param exploration_bias: float, optional. Bias term shifting metric to improvement over y_best + exploration_bias
    returns EI: ndarray (n_samples,) the expected improvement over y_best + exploration_bias for the input x
    """

    if len(x.shape) == 1:
        x = x.reshape((1, -1))

    y_best = y_best + exploration_bias

    mu, sigma = gp.predict(x, return_std=True)
    mu = mu.reshape(-1)
    Z = (mu-y_best)/sigma
    dist = norm(mu, sigma)
    EI = ((mu - y_best) * dist.cdf(Z) + sigma * dist.pdf(Z)).reshape(-1)
    EI[sigma < 0] = 0
    return EI


if __name__ == "__main__":
    from sklearn import gaussian_process
    np.random.seed(4)
    
    # search problem
    # domain_interval = (-1, 1)
    # range_interval = (0, 1)
    # def test_func(x):
    #     return 1-x**2

    # domain_interval = (-3, 10)
    # range_interval = (-0.5, 1)
    # def test_func(x):
    #     return np.sin(x-5)/(x-5+1e-7)

    domain_interval = (-3, 16)
    range_interval = (-0.5, 1)
    def test_func(x):
        return np.sin(x)

    noise_scale = 0.3
    noise_std = noise_scale * (range_interval[1] - range_interval[0])
    def observation_noise():
        return noise_std * np.random.randn()

    num_samples = 250

    # model
    kern = gaussian_process.kernels.WhiteKernel(noise_level_bounds=(1e-30, 100000))
    kern += gaussian_process.kernels.Matern(nu=3/2) * gaussian_process.kernels.ConstantKernel()
    gp = gaussian_process.GaussianProcessRegressor(kern, normalize_y=True, alpha=noise_std)


    exploration_bias = 0.1 * (range_interval[1] - range_interval[0])
    EI_exploration = partial(ExpectedImprovement, exploration_bias=exploration_bias)

    b = BayesianSearch(gp, EI_exploration, init_samples=10, num_restarts=1, 
                       x_dim=1, bounds=[domain_interval], tol=1e-9)

    for s in range(num_samples):
        x = b.get_next_sample()
        y = test_func(x) + observation_noise()
        print(f"Sample {s}: (x, y): ({x}, {y})")
        b.add_sample(x, y)
    
        # if s >= b.init_samples - 1:
        #     b.plot()
        #     x_plot = np.expand_dims(np.linspace(b.bounds[0][0], b.bounds[0][1], 200), -1)
        #     plt.plot(x_plot, test_func(x_plot), '-', label="True objective", c='c')
        #     plt.legend()
        #     plt.show()

    b.plot()
    x_plot = np.expand_dims(np.linspace(b.bounds[0][0], b.bounds[0][1], 200), -1)
    plt.plot(x_plot, test_func(x_plot), '-', label="True objective", c='c')
    plt.legend()
    plt.show()
# TODO

# needed to start testing in patients with tuning continuous parameters with BO, stim locations with statistical tests:
# checking against RAM implementation, particularly with their tests (which were minimal)
#       RAM implementation takes uniform samples if sample selected by Bayesian search is close to any previous sample
#       Tung's code for selecting the optimal parameters at a particular stim location doesn't use alpha for noise level, 
#       uses fit white kernel noise_level, which is added on top of fit kernel covariance? already included in kernel?
#       using white kernel noise_level as estimate of noise, rather than predictive sigma... 
#       is the predictive sigma not an estimate of the predictive uncertainty at a given point?
# 
#       sigma_cond = np.dot(k.T, np.linalg.inv(K+sigma_noise*np.eye(T,T)))
#       se = np.sqrt(np.dot(sigma_cond,sigma_cond.T)*sigma_noise)
#       wouldn't this tend to increase the sem with more samples?
#       
#       location selection:
#       t_stat_champ_sham = (champion_mean - sham_mean) / np.sqrt((champion_sem) ** 2 + (sham_sem) ** 2) 
#       p_val_champ_sham = norm.cdf(- np.abs(t_stat_champ_sham))
#       why take absolute value?
# 
#       otherwise, looks like the testing Tung did just involved a few functions with a few simple local optima
#       
# implementing in C++, being able to compare against Python outcomes
# testing algorithmic correctness, stability, and performance, find another reference implementation
#   difficult to test search against a reference implementation in terms of exact choices given chaos due
#   due to slightly different effects of noise/numerical differences
# estimating observation noise based on prior studies, ask Mike for data
# tuning hyperparameters based on test objectives

# choose optimal value as max_x GP(x)
# statistical tests for choosing between locations

# testing implementation correctness
# test acquisition function
#       test against reference implementation
# testing optimization of acquisition function
#       need to test number of restarts with global optimization algorithm
#       currently has high probability of returning different points across runs
#       frequently misses small peaky maxima
#       choose function with multiple local optima
#       try N local minima with basins occupying X% of the search space
#           compute probability of finding global optimum (or difference in objective from global optimum)
#           as a function of number of restarts/iterations
# then in addition to the GP tests, we've tested every component in our Bayesian search implementation

# performance testing: metrics used by Grace are good
#   select test functions, ideally would get Grace's code
#       multi-dimensional test functions needed
#   varying noise levels
#   varying acquisition functions and hyperparameters
#       probably just want to anneal an exploration-exploitation hyperparameter over training
#   varying GP kernels/HPs
#       try increasing
# don't need to tune super tightly given our test objective choices are ultimately arbitrary
#   mostly just trying to avoid severely suboptimal choices for HPs

# try increasing alpha, opening up kernel parameter bounds (maybe)
#       increasing alpha stopped GP from exact fitting to points
#       need to fit/estimate alpha without a priori knowledge of objective function range (Dan's project)
#       difference between white kernel and observation noise?

# controlling exploitation vs. exploration
#       add exploration hyperparameters in acquisition functions
#           need to wrap acquisition function in an object to vary exploration hps over search process
#           beyond this, we shouldn't do too much tuning, don't have a ton of data to evaluate algorithm performance
#               data is also extremely noisy, potentially much noisier than what I've already been tuning with

# get standard optimization test functions
# compare with PS4 code, maybe another implementation?
# implement in C++
# multidimensional testing with 2D color plotting
# testing

# LONG-TERM TODO
# implement some combo of GPR and GPC for discrete-continuous space, might be better to just use multi-armed bandits
#       probably better to just compare separate classes rather than make classes compete on the basis of limited samples
#       could potentially open up to a full multi-armed bandit selection scheme after sufficient samples selected in
#       each classes/stim. location

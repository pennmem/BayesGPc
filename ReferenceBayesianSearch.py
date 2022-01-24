import subprocess
import json
from tabnanny import verbose
from unittest.result import failfast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse
import logging
import os
from datetime import datetime
import time

from scipy import optimize

from skopt import Optimizer
from skopt import space
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process
from skopt.plots import plot_convergence
from skopt import acquisition

from sklearn import gaussian_process

from PyBenchFCN import Factory

GIT_BRANCH = subprocess.run(["git", "symbolic-ref", "HEAD"], capture_output=True, encoding="utf-8").stdout.strip('\n')
GIT_COMMIT = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, encoding="utf-8").stdout.strip('\n')
GIT_URL = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, encoding="utf-8").stdout.strip('\n')


# TODO add to PyBenchFCN, submit pull request
class hartmann4dfcn():
    def __init__(self, n_var=4):
        self.boundaries = np.array([0, 1])
        self.plot_bound = np.array([0, 1])
        if n_var != 4: print("Warning: hartmann4dfcn benchmark function is only defined for n_var==4. Overriding user input of ", n_var)
        self.n_var = 4
        # optimum value obtained with scipy.optimize.differential_evolution (default settings). 
        # Agreed with DE results from optim package with C++ implementation
        self.optimalX = np.array([0.18739526, 0.19415138, 0.55791773, 0.26477971])
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([[10,   3,   17,   3.5, 1.7, 8],
                           [0.05, 10,  17,   0.1, 8,   14],
                           [3,    3.5, 1.7,  10,  17,  8],
                           [17,   8,   0.05, 10,  0.1, 14]])
        self.P = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                           [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                           [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                           [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        self.optimalF = self.f(self.optimalX)

    def F(self, X):
        y = 1.1 * np.ones(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(4):
                inner = self.A[j, :4] * (X[i] - self.P[j, :4]) ** 2
                y[i] -= self.alpha[j] * np.exp(-inner.sum())
        y /= 0.839
        return y

    def f(self, x):
        return self.F(x[None, :]).item()


class TestFunction:
    def __init__(self, name, noise_level, x_dim, verbosity):
        self.name = name
        self.noise_level = noise_level
        self.x_dim = x_dim
        self.verbosity = verbosity
        if self.name == "sum_squares": self._benchmark_fcn = Factory.set_sop("sumsquares", self.x_dim)
        elif self.name == "hartmann4d": self._benchmark_fcn = hartmann4dfcn(self.x_dim)
        else: self._benchmark_fcn = Factory.set_sop(self.name, self.x_dim)
        self.x_interval = np.zeros((self.x_dim, 2))
        self.y_interval = np.zeros((2,))
        self._init()

    def _init(self):
        # x_dim, x_interval, y_interval set separately for each test function
        if self.name == "sin":
            raise NotImplementedError
            # gives two global maxima (peaks with equal heights)
            if self.x_dim != 1:
                print("Warning: 'sin' test function requires x_dim=1. Overriding user given value.")
                self.x_dim = 1
            self.x_interval[0, 0] = 3
            self.x_interval[0, 1] = 16
            self.y_interval[0] = -1
            self.y_interval[1] = 1
        
        elif self.name == "quadratic":
            raise NotImplementedError
            # simulates monotonic improvement to edge, testing for edge bias of global optimizer within BO
            for i in range(self.x_dim):
                self.x_interval[i, 0] = 0
                self.x_interval[i, 1] = 1
            self.y_interval[0] = 0
            self.y_interval[1] = 1

        elif self.name == "quadratic_over_edge":
            raise NotImplementedError
            # simulates monotonic improvement to edge, testing for edge bias of global optimizer within BO
            for i in range(self.x_dim):
                self.x_interval[i, 0] = 0
                self.x_interval[i, 1] = 1.5

            self.y_interval[0] = 0
            self.y_interval[1] = 1

        elif self.name == "PS4_1" or \
             self.name == "PS4_2" or \
             self.name == "PS4_3" or \
             self.name == "PS4_4":
            raise NotImplementedError
            # Test functions used in PS4
            if self.x_dim != 1:
                print("Warning: 'PS4_X' test functions require x_dim=1. Overriding user given value.")
                self.x_dim = 1

            for i in range(self.x_dim):
                self.x_interval[i, 0] = 0
                self.x_interval[i, 1] = 1

            # PS4 test functions. Some partial redundancy of functions. 
            # These variations on e.g. simple quadratics provide sanity
            # checks on scale invariance

            # medium-scale quadratic with negative range
            if self.name == "PS4_1":
                self.y_interval[0] = -0.515918
                self.y_interval[1] = -0.154138

            # multi-modal with different length scales of 
            # underlying functions: clear modes with long-term trend across domain
            elif self.name == "PS4_2":
                self.y_interval[0] = 0.211798
                self.y_interval[1] = 1.4019

            # small-scale quadratic
            elif self.name == "PS4_3":
                self.y_interval[0] = -0.000783627
                self.y_interval[1] = 0.101074
            # sin(10x)
            elif self.name == "PS4_4":
                self.y_interval[0] = -1
                self.y_interval[1] = 1

        # all standard optimization functions are ordinarily minimized. 
        # We negate these functions and negate and flip their global minima/maxima to allow for maximization

        # standard optimization test function. Highly multi-modal with
        # one global optimum. Used by Grace Dessert in her BO tests at Nia.
        elif self.name == "schwefel":
            self.x_interval[:, 0] = -500.0
            self.x_interval[:, 1] = 500.0

            if self.x_dim == 1: self.y_interval[0] = -837.966
            elif self.x_dim == 2: self.y_interval[0] = -1675.93
            elif self.x_dim == 3: self.y_interval[0] = -2513.9
            elif self.x_dim == 4: self.y_interval[0] = -3351.86
            elif self.x_dim == 5: self.y_interval[0] = -4189.83
            elif self.x_dim == 6: self.y_interval[0] = -4789.4
            else:
                raise ValueError("Min for test function " + self.name + " not yet added. " +
                                            "Please run TestFunction.get_func_optimum(true) and hardcode in the minimum value.")

            self.y_interval[1] = 0

        # standard multi-modal optimization test function.
        # Used by Grace Dessert in her BO tests at Nia.
        elif self.name == "hartmann4d":
            if self.x_dim != 4:
                print("Warning: 'hartmann4d' test function requires x_dim=4. Overriding user given value.")
                self.x_dim = 4
            for i in range(self.x_dim):
                self.x_interval[i, 0] = 0.0
                self.x_interval[i, 1] = 1.0
            self.y_interval[1] = 3.13449
            self.y_interval[0] = -1.31105

        # many local minima in larger shallow bowl
        elif self.name == "rastrigin":
            self.x_interval[:, 0] = -5.12
            self.x_interval[:, 1] = 5.12

            self.y_interval[1] = 0
            if self.x_dim == 1: self.y_interval[0] = -40.3533
            elif self.x_dim == 2: self.y_interval[0] = -80.7066
            elif self.x_dim == 3: self.y_interval[0] = -121.06
            elif self.x_dim == 4: self.y_interval[0] = -161.413
            elif self.x_dim == 5: self.y_interval[0] = -201.766
            elif self.x_dim == 6: self.y_interval[0] = -242.12
            else: raise ValueError("Test function lower range (y_interval(0)) not set for given input dimension x_dim")

        # many local minima in larger shallow bowl
        elif self.name == "ackley":
            self.x_interval[:, 0] = -32.768
            self.x_interval[:, 1] = 32.768

            self.y_interval[1] = 0
            if self.x_dim == 1: self.y_interval[0] = -22.3203
            elif self.x_dim == 2: self.y_interval[0] = -22.3203
            elif self.x_dim == 3: self.y_interval[0] = -22.3203
            elif self.x_dim == 4: self.y_interval[0] = -22.3203
            elif self.x_dim == 5: self.y_interval[0] = -22.3203
            elif self.x_dim == 6: self.y_interval[0] = -22.3187
            else: raise ValueError("Test function lower range (y_interval(0)) not set for given input dimension x_dim")

        # many local minima in larger shallow bowl
        elif self.name == "rosenbrock":
            self.x_interval[:, 0] = -5.0
            self.x_interval[:, 1] = 10.0

            self.y_interval[1] = 0
            if self.x_dim == 1: raise ValueError("Function 'rosenbrock' is degenerate for x_dim == 1")
            elif self.x_dim == 2: self.y_interval[0] = -1.10258e6
            elif self.x_dim == 3: self.y_interval[0] = -1.91266e6
            elif self.x_dim == 4: self.y_interval[0] = -2.72274e6
            elif self.x_dim == 5: self.y_interval[0] = -3.53282e6
            elif self.x_dim == 6: self.y_interval[0] = -4.3429e6
            else: raise ValueError("Test function lower range (y_interval(0)) not set for given input dimension x_dim. Please hard-code in TestFunction._init().")

        # many local minima in larger shallow bowl
        elif self.name == "eggholder":
            if self.x_dim != 2:
                print("Warning: 'eggholder' test function requires x_dim=2. Overriding user given value.")
                self.x_dim = 2

            for i in range(self.x_dim):
                self.x_interval[i, 0] = -512
                self.x_interval[i, 1] = 512
            self.y_interval[1] = 959.6407
            self.y_interval[0] = -1049.13  # approximate

        # many local minima in larger shallow bowl
        elif self.name == "sum_squares":
            self.y_interval[1] = 0
            self.y_interval[0] = 0
            for i in range(self.x_dim):
                self.x_interval[i, 0] = -10
                self.x_interval[i, 1] = 10
                self.y_interval[0] -= (i + 1) * self.x_interval[i, 1] * self.x_interval[i, 1]

        else:
            raise ValueError("Test function " + self.name + " not implemented." + 
                                        " Current options: " + 
                                        "'sin', " + 
                                        "'quadratic', " + 
                                        "'quadratic_over_edge', " + 
                                        "'PS4_1', " + 
                                        "'PS4_2', " + 
                                        "'PS4_3', " + 
                                        "'PS4_4', " +
                                        "'hartmann4d', " +
                                        "'schwefel', " +
                                        "'rastrigin', " +
                                        "'ackley', " +
                                        "'rosenbrock', " +
                                        "'sum_squares', " +
                                        "'eggholder', "
                                        )

        # negate y_interval for minimizing
        self.y_interval = -np.array([self.y_interval[1], self.y_interval[0]])

        assert self.x_dim > 0
        assert self.y_interval[1]>=self.y_interval[0]
        assert np.all(self.x_interval[:, 1]>=self.x_interval[:, 0])
        self.y_sol = self.y_interval[0]
        self.range = self.y_interval[1] - self.y_interval[0]
        self._noise_std = self.noise_level * self.range

    def f(self, x, add_noise=True):
        assert x.shape[-1] == self.x_dim, f"Input dimension != {self.x_dim}."
        assert len(x.shape) == 2
        y = self._benchmark_fcn.F(x)
        if add_noise:
            y += np.random.normal(0.0, self._noise_std, y.shape)

        return y

    def solution_error(self, x):
        error = (self.f(x, add_noise=False) - self.y_sol)/self.range
        return error

def test_BO(func_name, x_dim, args, full_metrics):
    fcn = TestFunction(func_name, args.noise_level, x_dim, args.verbosity)
    fd = func_name + ":d" + str(x_dim)
    full_metrics[fd] = dict()
    metrics = full_metrics[fd]
    verbose = args.verbosity

    x_bounds = [tuple(inter) for inter in fcn.x_interval]

    metrics["x_samples"] = []
    metrics["y_samples"] = []
    metrics["relative errors"] = []
    metrics["sample times"] = []

    max_pass_error = 0.05
    min_pass_prob = 0.9

    for run in range(args.n_runs):
        np.random.seed(args.seed + run)
        metrics["x_samples"].append([])
        metrics["y_samples"].append([])
        metrics["relative errors"].append([])
        metrics["sample times"].append([])

        # for matching CBayesianSearch
        # res = gp_minimize(f,                              # the function to minimize
        #                   x_bounds,                       # the bounds on each dimension of x
        #                   acq_func="EI",                  # the acquisition function
        #                   n_calls=n_iters,                # the number of evaluations of f
        #                   n_random_starts=n_init_samples, # the number of random initialization points
        #                   noise=0.5 * noise_level**2,     # the noise level (optional)
        #                   random_state=seed + run)        # the random seed


        # set up scikit-opt BO search space
        # search space for ray-tracing problem
        # search_space = [
        #     space.Categorical((0, 1, 2, 3), transform='normalize'),  # Light identifier
        #     space.Real(-100, 100, transform='normalize'),  # Light's X coordinate
        #     space.Real(-100, 100, transform='normalize'),  # Light's Y coordinate
        #     space.Real(-100, 100, transform='normalize')]  # Light's Z coordinate
        # search_space = [space.Real(x_bounds[d][0], x_bounds[d][1], transform='normalize') for d in range(x_dim)]  # Light's Z coordinate
        search_space = [space.Real(x_bounds[d][0], x_bounds[d][1], transform='identity') for d in range(x_dim)]
        transformed_x_bounds = [s.transformed_bounds for s in search_space]

        kern = gaussian_process.kernels.WhiteKernel(noise_level=1.0,
                                                    noise_level_bounds=(1e-6, 1000))
        mean_domain_length = np.diff(fcn.x_interval, axis=1).mean()
        kern += gaussian_process.kernels.Matern(nu=3/2, 
                                                length_scale=1.0,
                                                length_scale_bounds=(0.05*mean_domain_length, 
                                                                     2.0*mean_domain_length))\
                * gaussian_process.kernels.ConstantKernel(constant_value=1.0,
                                                          constant_value_bounds=(0.5, 2.0))
        # observation noise not currently being internally scaled with standard deviation of samples... (standard sklearn implementation)
        gp = gaussian_process.GaussianProcessRegressor(kern, normalize_y=True, alpha=(0.5*fcn.noise_level) ** 2)

        exp_bias = args.exp_bias * fcn.range

        opt = Optimizer(search_space, 
                        gp, 
                        acq_func="EI", 
                        acq_optimizer="sampling", 
                        n_initial_points=args.n_init_samples, 
                        random_state=args.seed + run, 
                        initial_point_generator='random',
                        acq_func_kwargs={"xi": exp_bias})

        for iter in range(args.n_iters):
            t = time.time()
            x = opt.ask()
            x = np.array(x).reshape(1, -1)
            y = fcn.f(x)
            temp = f"Sample {iter + 1}: (x, y) ({x}, {y})"
            if verbose > 0:
                print(temp)
            logging.info(temp)
            x = list(x[0])
            y = float(y)
            res = opt.tell(x, y)
            metrics["sample times"][run].append(time.time() - t)
            metrics["x_samples"][run].append(x)
            metrics["y_samples"][run].append([y])

            if (iter == args.n_iters - 1) or args.full_errors:
                res_temp = optimize.differential_evolution(
                    lambda x: opt.models[-1].predict(x.reshape(1, -1)),
                    transformed_x_bounds,
                    maxiter=args.x_dim * 500,
                    popsize=args.x_dim * 300)  # consistently tuned all benchmark functions (except for eggholder, which was close)
                x_best = res_temp.x
                x_best = [s.inverse_transform(x_best[i])
                          for i, s in enumerate(search_space)]
                x_best = np.asarray(x_best).reshape(1, -1)
                metrics["relative errors"][run].append(float(fcn.solution_error(x_best)))
                if verbose > 0:
                    print(f"x-best: {x_best}, y-best: {fcn.f(x_best, add_noise=False)}, Rel error: {fcn.solution_error(x_best)}")

            # plotting
            if args.x_dim < 3 and args.plot and ((iter == args.n_iters - 1) or (verbose > 1)):
                plt.figure()
                if x_dim == 1:
                    show_legend = args.n_iters == 0
                    _ = plot_gaussian_process(res, objective=lambda x: fcn.f(np.array(x).reshape(1, -1), add_noise=False),
                                            noise_level=args.noise_level,
                                            show_acq_func=True)
                elif x_dim == 2:
                    n_points = 150
                    levels = 20
                    x1s = np.linspace(fcn.x_interval[0, 0], fcn.x_interval[0, 1], n_points)
                    x2s = np.linspace(fcn.x_interval[1, 0], fcn.x_interval[1, 1], n_points)
                    x1, x2 = np.meshgrid(x1s, x2s)
                    X = np.concatenate([x1.reshape(-1, 1), 
                                        x2.reshape(-1, 1)], 
                                        axis=1)
                    f_true = fcn.f(X, add_noise=False).reshape(x1.shape)
                    # f_min = np.min(f_true)
                    # f_max = np.max(f_true)
                    ax = plt.axes(projection='3d')
                    ax.plot_surface(x1, x2, f_true, cmap='viridis', edgecolor='none')
                    ax.set_title(f"True function: {args.func}")
                    plt.figure()
                    cs = plt.contourf(x1, x2, f_true, levels=levels)
                    plt.colorbar(cs)
                    plt.scatter(np.array(opt.Xi)[:, 0], 
                                np.array(opt.Xi)[:, 1], s=1, marker='.', c="k")
                    plt.title(f"True function: {args.func}")

                    X_transformed = np.concatenate([s.transform(X[:, i]).reshape(-1, 1) for i, s in enumerate(search_space)], axis=1)
                    mu, std = opt.models[-1].predict(X_transformed, return_std=True)
                    mu = mu.reshape(x1.shape)
                    std = std.reshape(x1.shape)
                    plt.figure()
                    cs = plt.contourf(x1, x2, mu, levels=levels)
                    plt.colorbar(cs)
                    plt.scatter(np.array(opt.Xi)[:, 0], 
                                np.array(opt.Xi)[:, 1], s=1, marker='.', c="k")
                    plt.title(f"GP mean")

                    plt.figure()
                    cs = plt.contourf(x1, x2, std, levels=levels)
                    plt.colorbar(cs)
                    plt.scatter(np.array(opt.Xi)[:, 0], 
                                np.array(opt.Xi)[:, 1], s=1, marker='.', c="k")
                    plt.title(f"GP std")

                    acq = acquisition._gaussian_acquisition(X_transformed, opt.models[-1],
                            y_opt=np.min(opt.yi),
                            acq_func=opt.acq_func,
                            acq_func_kwargs=opt.acq_func_kwargs).reshape(x1.shape)

                    plt.figure()
                    cs = plt.contourf(x1, x2, acq, levels=levels)
                    plt.colorbar(cs)
                    plt.scatter(np.array(opt.Xi)[:, 0],
                                np.array(opt.Xi)[:, 1], s=1, marker='.', c="k")
                    plt.title(f"Acquisition function")
                else:
                    raise ValueError
                print(opt.models[-1].kernel_)
                plt.show()
        
        logging.info(f"Run {run} time: " + str(np.sum(metrics["sample times"][-1])))

        # the approximated minimum is found to be:
        # "x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)

        #############################################################################
        # For further inspection of the results, attributes of the `res` named tuple
        # provide the following information:
        #
        # - `x` [float]: location of the minimum.
        # - `fun` [float]: function value at the minimum.
        # - `models`: surrogate models used for each iteration.
        # - `x_iters` [array]:
        #   location of function evaluation for each iteration.
        # - `func_vals` [array]: function value for each iteration.
        # - `space` [Space]: the optimization space.
        # - `specs` [dict]: parameters passed to the function.

    fail = 0
    return fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="skopt")
    parser.add_argument("--logdir", type=str, default="results")
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--full_errors", action="store_true")
    parser.add_argument("--full_time_test", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--func", type=str, default="all")
    parser.add_argument("--x_dim", type=int, default=1)

    # model hyperparameters
    parser.add_argument("--kern", type=str, default="Matern32")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_iters", type=int, default=250)
    parser.add_argument("--n_init_samples", type=int, default=50)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--exp_bias", type=float, default=0.25)


    args = parser.parse_args()

    if args.tag != "skopt":
        raise NotImplementedError

    args.system = "python"
    args.GIT_BRANCH = GIT_BRANCH
    args.GIT_COMMIT = GIT_COMMIT
    args.GIT_URL = GIT_URL

    def getDateTime():
        return datetime.now().strftime("%m-%d-%y_%H:%M:%S")

    args.datetime = getDateTime()

    logdir = os.path.join(args.logdir, args.tag + \
                            f"-func_{args.func}" + \
                            f"-dim_{args.x_dim}" + \
                            f"-kern_{args.kern}" + \
                            f"-runs_{args.n_runs}" + \
                            f"-iters_{args.n_iters}" + \
                            f"-init_samp_{args.n_init_samples}" + \
                            f"-exp_bias_{args.exp_bias}" + \
                            "_" + getDateTime())
    os.makedirs(logdir)
    print("Making log directory:\n" + logdir)
    logging.info("Making log directory: " + logdir)


    with open(os.path.join(logdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    # human-readable log
    logging.basicConfig(filename=os.path.join(logdir, 'log.out'), level=logging.DEBUG)
    
    # metrics dictionary to be dumped into JSON for easy parsing
    metrics = dict()

    if args.func == "all":
        funcs = [#("sin", 1),
                #  ("quadratic", 1),
                #  ("quadratic_over_edge", 1),
                #  ("PS4_1", 1),
                #  ("PS4_2", 1),
                #  ("PS4_3", 1),
                #  ("PS4_4", 1),
                # ("schwefel", 1),
                #  ("schwefel", 2),
                #  ("schwefel", 4),
                # ("hartmann4d", 4),
                # ("ackley", 1),
                #  ("ackley", 2),
                #  ("ackley", 4),
                # ("rastrigin", 1),
                #  ("rastrigin", 2),
                #  ("rastrigin", 4),
                ("eggholder", 2),
                # ("sum_squares", 1),
                #  ("sum_squares", 2),
                #  ("sum_squares", 4),
                # ("rosenbrock", 2),
                #  ("rosenbrock", 4),
        ]
    else:
        funcs = [(args.func, args.x_dim)]

    n_failures = 0
    for fcn, x_dim in funcs:
        # test global optimizer
        # np.random.seed(args.seed)
        # f = TestFunction(fcn, args.noise_level, x_dim, args.verbosity)
        # res = optimize.differential_evolution(
        #     lambda x: f.f(x.reshape(1, -1), add_noise=False),
        #     [tuple(inter) for inter in f.x_interval],
        #     maxiter=args.x_dim * 500,
        #     popsize=args.x_dim * 300)
        # x_best = res.x
        # print(f"Function: {fcn} x_dim: {x_dim}\n\tx-best: {x_best}\n\ty-best: {f.f(x_best.reshape(1, -1), add_noise=False)}\n\ty-sol: {f.y_sol}")

        fails_test = test_BO(fcn, x_dim, args, metrics)
        n_failures += fails_test
    logging.info(f"Number of test failures: {n_failures}/{len(funcs)}")
    
    with open(os.path.join(logdir, "log.json"), "w") as f:
        try:
            json.dump(metrics, f, indent=1)
        except Exception as e:
            logger.debug(f"Failed to write output to JSON. Error code:\n{e}")

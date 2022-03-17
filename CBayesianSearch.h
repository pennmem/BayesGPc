#ifndef BAYESIANSEARCH_H
#define BAYESIANSEARCH_H

#define OPTIM_ENABLE_EIGEN_WRAPPERS
// #define OPTIM_ENABLE_ARMA_WRAPPERS
// #define ARMA_DONT_USE_WRAPPER
#include "optim.hpp"

// enables access to private variables in CGp, not ideal but using for now...
// pkern accessed in CBayesianSearch.cpp
#define DBG

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include <cmath>
#include <cassert>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <stdexcept>

#define _NDLCASSERT

class BayesianSearchModel {
    public:
        BayesianSearchModel() {}
        BayesianSearchModel(CCmpndKern& kernel, const CMatrix& param_bounds,
                            double observation_noise, double exp_bias,
                            int init_samples, int rng_seed, int verbose) {
            kern = kernel.clone();
            DIMENSIONMATCH(param_bounds.getCols() == 2);
            bounds = param_bounds;
            seed = rng_seed;
            verbosity = verbose;
            _init();
            x_dim = bounds.getRows();
            num_samples = 0;
            initial_samples = init_samples;
            obsNoise = observation_noise;
            exploration_bias = exp_bias;
            acq_func_name = "expected_improvement";
            y_best = -INFINITY;
            x_samples = new CMatrix(1, x_dim);
            y_samples = new CMatrix(1, 1);
        }

        // explicit copy constructor to force compiler to not apply implicit move operations
        BayesianSearchModel(const BayesianSearchModel& bay) {
            kern = bay.kern->clone();
            DIMENSIONMATCH(bay.bounds.getCols() == 2);
            bounds = bay.bounds;
            seed = bay.seed;
            verbosity = bay.verbosity;
            _init();
            x_dim = bay.x_dim;
            num_samples = 0;
            initial_samples = bay.initial_samples;
            obsNoise = bay.obsNoise;
            exploration_bias = bay.exploration_bias;
            acq_func_name = "expected_improvement";
            y_best = -INFINITY;
            x_samples = new CMatrix(1, x_dim);
            y_samples = new CMatrix(1, 1);
        }

        ~BayesianSearchModel() {
            if (x_samples) { delete x_samples; }
            if (y_samples) { delete y_samples; }
        }

        void _init() {
            if (seed != -1) {
                rng.seed(seed);
            }
        }

        int num_samples;
        int x_dim;
        int initial_samples;
        int seed;
        int verbosity;
        boost::mt19937 rng;
        CMatrix* x_samples;
        CMatrix* y_samples;
        double y_best;
        // observation noise variance. Currently applied to only training samples.
        double obsNoise;
        double exploration_bias;

        // CGp model
        CCmpndKern* kern;
        CGaussianNoise* noiseInit = nullptr;
        CGp* gp = nullptr;

        CMatrix bounds;
        CMatrix* get_next_sample();
        void add_sample(const CMatrix& x, const CMatrix& y);
        CMatrix* get_best_solution();
        // TODO switch to enumeration
        string acq_func_name;
        // double expected_improvement(const CMatrix* x, const CGp model, double y_b, double exp_bias);
};

double expected_improvement(const CMatrix& x, const CGp& model, double y_b, double exp_bias);

#endif

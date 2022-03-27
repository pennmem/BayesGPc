#ifndef BAYESIANSEARCH_H
#define BAYESIANSEARCH_H

#ifndef _WIn

#define OPTIM_ENABLE_EIGEN_WRAPPERS
// #define OPTIM_ENABLE_ARMA_WRAPPERS
// #define ARMA_DONT_USE_WRAPPER
#include "optim.hpp"

#endif

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

double expected_improvement(const CMatrix& x, const CGp& model, double y_b, double exp_bias);

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
            // hard code expected improvement for now
            acq_fcn = [&](const CMatrix& x) {
                    return expected_improvement(x, *(this->gp), this->y_best, this->exploration_bias); };
            y_best = -INFINITY;
            x_samples = new CMatrix(1, x_dim);
            y_samples = new CMatrix(1, 1);
            int n_grid;
            if (x_dim == 1) { n_grid = 50000; }
            else if (x_dim == 2) { n_grid = 10000; }
            else if (x_dim == 3) { n_grid = 1000; }
            else if (x_dim == 4) { n_grid = 1000; }
            else { throw std::runtime_error("Need to hard code n_grid for x_dim >= 5"); }

            for (int i = 0; i < x_dim; i++) {
                CMatrix grid1D = linspace(bounds.getVal(i,0),
                                          bounds.getVal(i,1),
                                          n_grid);
                grid_vals.push_back(grid1D);
            }
        }

        // explicit copy constructor to force compiler to not apply implicit move operations
        BayesianSearchModel(const BayesianSearchModel& bay) {
            kern = bay.kern->clone();
            DIMENSIONMATCH(bay.bounds.getCols() == 2);
            bounds = bay.bounds;
            grid_vals = bay.grid_vals;
            seed = bay.seed;
            verbosity = bay.verbosity;
            _init();
            x_dim = bay.x_dim;
            num_samples = 0;
            initial_samples = bay.initial_samples;
            obsNoise = bay.obsNoise;
            exploration_bias = bay.exploration_bias;
            acq_func_name = "expected_improvement";
            acq_fcn = [&](const CMatrix& x) {
                    return expected_improvement(x, *(this->gp), this->y_best, this->exploration_bias); };
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
        // for grid search-based global optimization
        std::vector<CMatrix> grid_vals;
        CMatrix* get_next_sample();
        void add_sample(const CMatrix& x, const CMatrix& y);
        CMatrix* get_best_solution();
        // TODO switch to enumeration
        string acq_func_name;
        std::function<double(const CMatrix&)> acq_fcn;
        // double expected_improvement(const CMatrix* x, const CGp model, double y_b, double exp_bias);
};

#endif

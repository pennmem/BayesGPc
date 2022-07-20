#ifndef BAYESIANSEARCH_H
#define BAYESIANSEARCH_H

#ifdef INCLUDE_OPTIM
#ifndef _WIN
#define OPTIM_ENABLE_EIGEN_WRAPPERS
// #define OPTIM_ENABLE_ARMA_WRAPPERS
// #define ARMA_DONT_USE_WRAPPER
#include "optim.hpp"
#endif  // ifndef _WIN
#endif  // INCLUDE_OPTIM

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include "CNoise.h"
#include <cmath>
#include <cassert>

#include <boost/math/distributions/normal.hpp>

#include <stdexcept>
#include <nlohmann/json.hpp>

#define _NDLCASSERT

double expected_improvement(const CMatrix& x, const CGp& model, double y_b, double exp_bias);

class BayesianSearchModel {
    public:
        BayesianSearchModel() {}
        BayesianSearchModel(CCmpndKern& kernel, const CMatrix& param_bounds,
                            double observation_noise, double exp_bias,
                            int init_samples, int rng_seed, int verbose, vector<CMatrix> grid=vector<CMatrix>()) {
            kern.reset(kernel.clone());
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

            x_samples.reset(new CMatrix(1, x_dim));
            y_samples.reset(new CMatrix(1, 1));
            grid_vals = grid;
            #ifdef INCLUDE_OPTIM
            if (grid.size() == 0) {
                #ifdef _WIN 
                throw std::exception("Windows build only supports grid search. Please provide parameter [grid].");
                #else
                optimization_fcn = "de";
                #endif  // ifndef _WIN
            }
            #endif  // INCLUDE_OPTIM
        }

        // explicit copy constructor to force compiler to not apply implicit move operations
        BayesianSearchModel(const BayesianSearchModel& bay) {
            kern.reset(bay.kern->clone());
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
            x_samples.reset(new CMatrix(1, x_dim));
            y_samples.reset(new CMatrix(1, 1));
        }

        void _init() {
            if (seed != -1) {
                set_seed(seed);
            }
        }
        void set_seed(int val) {
            seed = val;
            rng.seed(seed);
        }

        // CGp model
        std::unique_ptr<CGp> gp;
        // #ifndef DEBUG
        // private:
        // #endif
        std::unique_ptr<CCmpndKern> kern;
        std::unique_ptr<CGaussianNoise> noiseInit;

        int seed;
        int num_samples;
        int x_dim;
        int initial_samples;
        int verbosity;
        std::mt19937 rng;
        std::unique_ptr<CMatrix> x_samples;
        std::unique_ptr<CMatrix> y_samples;
        double y_best;
        // observation noise variance. Currently applied to only training samples.
        double obsNoise;
        double exploration_bias;

        CMatrix bounds;
        // for grid search-based global optimization
        string optimization_fcn = "grid";
        bool init_points_on_grid = false;
        std::vector<CMatrix> grid_vals;
        CMatrix get_next_sample();
        void add_sample(const CMatrix& x, const CMatrix& y);
        void updateGP();
        CMatrix get_best_solution();
        CMatrix uniform_random_sample();
        // TODO switch to enumeration
        string acq_func_name;
        std::function<double(const CMatrix&)> acq_fcn;
        // double expected_improvement(const CMatrix& x, const CGp model, double y_b, double exp_bias);
        json json_structure() const;
        json json_state() const;
};

#endif

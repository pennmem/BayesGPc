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
        BayesianSearchModel();
        BayesianSearchModel(CCmpndKern& kernel, const CMatrix& param_bounds,
                            double observation_noise, double exp_bias,
                            size_t init_samples, int rng_seed, int verbose, vector<CMatrix> grid=vector<CMatrix>());
        // explicit copy constructor to force compiler to not apply implicit move operations
        BayesianSearchModel(const BayesianSearchModel& bay);
        void operator=(const BayesianSearchModel& bay);
        ~BayesianSearchModel();
        void _init();
        void set_seed(int val);

        // CGp model
        std::unique_ptr<CGp> gp;
        // #ifndef DEBUG
        // private:
        // #endif
        std::unique_ptr<CCmpndKern> kern;
        std::unique_ptr<CGaussianNoise> noiseInit;

        size_t num_samples;
        size_t x_dim;
        size_t initial_samples;
        int seed;
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

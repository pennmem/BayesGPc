#define OPTIM_ENABLE_EIGEN_WRAPPERS
// #define OPTIM_ENABLE_ARMA_WRAPPERS
// #define ARMA_DONT_USE_WRAPPER
#include "optim.hpp"

#include "CGp.h"
#include "CKern.h"
#include "CMatrix.h"
#include <cmath>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <stdexcept>


class BayesianSearchModel {
    public:
        BayesianSearchModel(CCmpndKern& kernel, CMatrix* param_bounds, double exp_bias, int init_samples, int rng_seed) {
            kern = kernel;
            // TODO check bounds dimensions
            bounds = param_bounds;
            seed = rng_seed;
            _init();
            x_dim = bounds->getRows();
            num_samples = 0;
            initial_samples = init_samples;
            exploration_bias = exp_bias;
            acq_func_name = "expected_improvement";
            y_best = -INFINITY;
            x_samples = new CMatrix(1, x_dim);
            y_samples = new CMatrix(1, 1);
        }

        ~BayesianSearchModel() {
            delete x_samples;
            delete y_samples;
        }

        void _init() {
            if (seed != -1) {
                rng.seed(seed);
            }
        }

        // TODO use smart pointers?
        int num_samples;
        int x_dim;
        int initial_samples;
        int seed;
        boost::mt19937 rng;
        CMatrix* x_samples;
        CMatrix* y_samples;
        double y_best;
        double exploration_bias;
        CCmpndKern kern;
        CGp gp;
        CMatrix* bounds;
        CMatrix* get_next_sample();
        void add_sample(const CMatrix& x, const CMatrix& y);
        // TODO switch to enumeration
        string acq_func_name;
        // double expected_improvement(const CMatrix* x, const CGp model, double y_b, double exp_bias);
};

double expected_improvement(const CMatrix& x, const CGp& model, double y_b, double exp_bias);

// TODO
// design wrapper around GCp, should really just have sklearn API
// clean up

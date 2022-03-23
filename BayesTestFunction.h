#ifndef BAYESTESTFUNCTION_H
#define BAYESTESTFUNCTION_H

#define OPTIM_ENABLE_EIGEN_WRAPPERS
// #define OPTIM_ENABLE_ARMA_WRAPPERS
// #define ARMA_DONT_USE_WRAPPER
#include "optim.hpp"

#include <string>
#include "CKern.h"
#include "CMatrix.h"

#ifdef _WIN
#include "ndlutil.h"
#endif

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>


class TestFunction {
  public:
    string name;
    int seed = 1235;
    default_random_engine e;
    int x_dim;
    CMatrix x_interval;
    CMatrix y_interval;
    double range;
    double noise_level = 0.3;
    double noise_std;
    double y_sol;
    int verbosity;

    // noise distribution
    normal_distribution<double> dist;

    TestFunction(string func_name, 
                 int seed_temp, 
                 double noise_level_temp, 
                 int x_dim_temp=1,
                 int verbose=0);
    // TestFunction(string func_name, 
    //              int seed_temp, 
    //              double noise_level_temp, 
    //              int x_dim_temp=1,
    //              int verbose=0) {
    //   seed = seed_temp;
    //   noise_level = noise_level_temp;
    //   name = func_name;
    //   verbosity = verbose;
    //   x_dim = x_dim_temp;
    //   default_random_engine e(seed);
    //   _init();
    //   noise_std = noise_level * range;
    //   dist = normal_distribution(0.0, noise_std);
    // }

    struct funcOptimStruct {
      TestFunction* test;
      bool noise;
      bool neg = true;
    };

    void _init();
    CMatrix func(const CMatrix& x, bool add_noise=true);
    double solution_error(const CMatrix& x_best);
    double func_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data);
    CMatrix* get_func_optimum(bool get_min = true);
    void reseed(int i);
};

#endif

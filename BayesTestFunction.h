#ifndef BAYESTESTFUNCTION_H
#define BAYESTESTFUNCTION_H

#ifdef INCLUDE_OPTIM
#ifndef _WIN
#define OPTIM_ENABLE_EIGEN_WRAPPERS
// #define OPTIM_ENABLE_ARMA_WRAPPERS
// #define ARMA_DONT_USE_WRAPPER
#include "optim.hpp"
#endif  // ifndef _WIN
#endif  // INCLUDE_OPTIM

#include <string>
#include "CKern.h"
#include "CMatrix.h"

#ifdef _WIN
#include "ndlutil.h"
#endif

#include <random>

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
    std::vector<CMatrix> grid_vals;
    int verbosity;
    // "grid" for grid search, "de" for differential evolution (optim implementation; requires #define INCLUDE_OPTIM)
    string optimization_fcn = "grid";

    // noise distribution
    normal_distribution<double> dist;

    TestFunction(string func_name, 
                 int seed_temp, 
                 double noise_level_temp, 
                 int x_dim_temp=1,
                 int verbose=0);

    struct funcOptimStruct {
      TestFunction* test;
      bool noise;
      bool neg = true;
    };

    void _init();
    CMatrix func(const CMatrix& x, bool add_noise=true);
    double solution_error(const CMatrix& x_best);

    #ifdef INCLUDE_OPTIM
    #ifndef _WIN
    double func_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data);
    #endif  // ifndef _WIN
    #endif  // INCLUDE_OPTIM
    CMatrix* get_func_optimum(bool get_min = true);
    void set_seed(int i);
    int get_seed();
};

#endif

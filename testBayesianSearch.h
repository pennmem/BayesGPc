#ifndef TESTBAYESIANSEARCH_H
#define TESTBAYESIANSEARCH_H

#include <string>
#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"
#include "CNdlInterfaces.h"
#include "Logger.h"
#include "CBayesianSearch.h"
#ifdef _WIN
#include <QDir>
#else
#include "bayesPlotUtil.h"
#endif
#include "BayesTestFunction.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <ctime>
#include <random>
#include <cassert>
#include "version.h"

int testBayesianSearch(CML::EventLog& log,
                       string& json_dir,
                       string kernel, 
                       string test_func_str,
                       int n_runs=25,
                       int n_iters=250,
                       int n_init_samples=10,
                       int x_dim=1,
                       int n_grid=0,
                       double noise_level=0.3,
                       double lengthscale_lb=0.1,
                       double lengthscale_ub=2.0,
                       double white_lb=0.1,
                       double white_ub=4.0,
                       double exp_bias_ratio=0.1,
                       int verbosity=0,
                       bool full_time_test=false,
                       bool plotting=false,
                       int seed=1234);

CCmpndKern getTestKernel(const string kernel, 
                         const TestFunction fcn,
                         double lengthscale_lb=0.1,
                         double lengthscale_ub=2.0,
                         double white_lb=0.1,
                         double white_ub=4.0
                         )
{
    // take mean across all dimension ranges to obtain univariate length scale range
    // For now, all dimension ranges are the same, and as long as they are, this choice 
    // is fine.
    CMatrix mean_x_interval(meanCol(fcn.x_interval));
    double x_range = mean_x_interval.getVal(1) - mean_x_interval(0);
    assert(x_range >= 0);
    int x_dim = fcn.x_dim;
    // TODO put in separate function here and in testBayesianSearch
    CKern* k = kernelFactory(kernel, x_dim);
    CCmpndKern kern(x_dim);
    kern.addKern(k);
    CWhiteKern* whitek = new CWhiteKern(x_dim);
    kern.addKern(whitek);
    delete k;
    delete whitek;
    
    // set kernel hyperparameter bounds
    // no meaningful bounds on interpolation variance for now, might want min var
    double var_lb = 0.25;
    double var_ub = 4.0;
    double lengthScale_lb = lengthscale_lb * x_range;
    double lengthScale_ub = lengthscale_ub * x_range;
    CMatrix b(1, 2);
    if (kernel.compare("Matern32") == 0) {
    b(0, 0) = lengthScale_lb;
    b(0, 1) = lengthScale_ub;
    kern.setBoundsByName("matern32_0__lengthScale", b);
    b(0, 0) = var_lb;
    b(0, 1) = var_ub;
    kern.setBoundsByName("matern32_0__variance", b);
    }
    else if (kernel.compare("Matern52") == 0) {
    b(0, 0) = lengthScale_lb;
    b(0, 1) = lengthScale_ub;
    kern.setBoundsByName("matern52_0__lengthScale", b);
    b(0, 0) = var_lb;
    b(0, 1) = var_ub;
    kern.setBoundsByName("matern52_0__variance", b);
    }
    else if (kernel.compare("RBF") == 0) {
    // squared reciprocal "length scale"
    b(0, 0) = 1/(lengthScale_ub * lengthScale_ub);
    b(0, 1) = 1/(lengthScale_lb * lengthScale_lb);
    kern.setBoundsByName("rbf_0__inverseWidth", b);
    b(0, 0) = var_lb;
    b(0, 1) = var_ub;
    kern.setBoundsByName("rbf_0__variance", b);
    }
    else if (kernel.compare("RationalQuadratic") == 0) {
    b(0, 0) = lengthScale_lb;
    b(0, 1) = lengthScale_ub;
    kern.setBoundsByName("ratquad_0__lengthScale", b);
    b(0, 0) = 0.1;
    b(0, 1) = 10.0;
    kern.setBoundsByName("ratquad_0__alpha", b);
    b(0, 0) = var_lb;
    b(0, 1) = var_ub;
    kern.setBoundsByName("ratquad_0__variance", b);
    }
    b(0, 0) = 1e-4 + white_lb; //* fcn.noise_level * fcn.noise_level;  //0.001;
    b(0, 1) = white_ub;
    kern.setBoundsByName("white_1__variance", b);
    return kern;
}

#endif

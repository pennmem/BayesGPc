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
                       json& json_log,
                       string kernel, 
                       string test_func_str,
                       int n_runs=25,
                       int n_iters=250,
                       int n_init_samples=10,
                       int x_dim=1,
                       double noise_level=0.3,
                       double exp_bias_ratio=0.1,
                       int verbosity=0,
                       bool full_time_test=false,
                       bool plotting=false,
                       int seed=1234);

CCmpndKern getTestKernel(const string kernel, const double range, const int x_dim)
{
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
    CMatrix b(1, 2);
    if (kernel.compare("Matern32") == 0) {
    b(0, 0) = 0.1 * range;
    b(0, 1) = 2.0 * range;
    kern.setBoundsByName("matern32_0:lengthScale", b);
    b(0, 0) = 0.25;
    b(0, 1) = 4.0;
    kern.setBoundsByName("matern32_0:variance", b);
    }
    else if (kernel.compare("Matern52") == 0) {
    b(0, 0) = 0.1 * range;
    b(0, 1) = 2.0 * range;
    kern.setBoundsByName("matern52_0:lengthScale", b);
    b(0, 0) = 0.25;
    b(0, 1) = 4.0;
    kern.setBoundsByName("matern52_0:variance", b);
    }
    else if (kernel.compare("RBF") == 0) {
    // squared reciprocal "length scale"
    b(0, 0) = 1/(4.0 * range*range);
    b(0, 1) = 1/(0.01 * range*range);
    kern.setBoundsByName("rbf_0:inverseWidth", b);
    b(0, 0) = 0.25;
    b(0, 1) = 4.0;
    kern.setBoundsByName("rbf_0:variance", b);
    }
    else if (kernel.compare("RationalQuadratic") == 0) {
    b(0, 0) = 0.1 * range;
    b(0, 1) = 2.0 * range;
    kern.setBoundsByName("ratquad_0:lengthScale", b);
    b(0, 0) = 0.1;
    b(0, 1) = 10.0;
    kern.setBoundsByName("ratquad_0:alpha", b);
    b(0, 0) = 0.25;
    b(0, 1) = 4.0;
    kern.setBoundsByName("ratquad_0:variance", b);
    }
    b(0, 0) = 0.001;
    b(0, 1) = 4.0;
    kern.setBoundsByName("white_1:variance", b);
    return kern;
}

#endif

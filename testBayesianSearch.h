#ifndef TESTBAYESIANSEARCH_H
#define TESTBAYESIANSEARCH_H

#include <string>
#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"
#include "cnpy.h"
#include "sklearn_util.h"
#include "CNdlInterfaces.h"
#include "Logger.h"
#include "CBayesianSearch.h"
#include "bayesPlotUtil.h"
#include "BayesTestFunction.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <ctime>
#include <random>
#include <cassert>


int testBayesianSearch(CML::EventLog& log,
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
                       bool plotting=false);

#endif

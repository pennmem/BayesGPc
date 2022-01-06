#ifndef BAYESPLOTUTIL_H
#define BAYESPLOTUTIL_H

#include "CBayesianSearch.h"
#include <matplot/matplot.h>
#include "CMatrix.h"
#include <thread>
using namespace matplot;

void plot_BO_state(const BayesianSearchModel& BO, 
                   const CMatrix& x_plot, const CMatrix& y_plot,
                   const CMatrix& y_pred, const CMatrix& std_pred, 
                   CMatrix* x_sample, CMatrix* y_sample);

void plot_BO_sample_times(const CMatrix sample_times, const string func_name);

#endif

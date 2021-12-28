#include "bayesPlotUtil.h"

void plot_BO_state(const BayesianSearchModel& BO, const CMatrix& x_plot, const CMatrix& y_plot, 
                   const CMatrix& y_pred, const CMatrix& std_pred, 
                   CMatrix* x_sample, CMatrix* y_sample) {
  // plotting
  int x_dim = BO.x_dim;
  // 1D plotting
  assert(x_dim == 1 && x_plot.getCols()==1);
  int n_plot = x_plot.getRows();
  CMatrix* x_sample_plot = x_sample;
  CMatrix* y_sample_plot = y_sample;
  vector<double> x_vec(x_plot.getVals(), x_plot.getVals() + n_plot);
  vector<double> y_true_vec(y_plot.getVals(), y_plot.getVals() + n_plot);
  std::vector<double> x_samples_vec(BO.x_samples->getVals(), BO.x_samples->getVals() + BO.num_samples);
  std::vector<double> y_samples_vec(BO.y_samples->getVals(), BO.y_samples->getVals() + BO.num_samples);
  std::vector<double> y_pred_vec(y_pred.getVals(), y_pred.getVals() + n_plot);
  std::vector<double> y_pred_plus_std(y_pred_vec);
  std::vector<double> y_pred_minus_std(y_pred_vec);
  std::vector<double> std_pred_vec(std_pred.getVals(), std_pred.getVals() + n_plot);
  std::vector<double> x_new_sample_vec(x_sample_plot->getVals(), x_sample_plot->getVals() + 1);
  std::vector<double> y_new_sample_vec(y_sample_plot->getVals(), y_sample_plot->getVals() + 1);
  CMatrix acq_func_plot(n_plot, 1);

  for (int i = 0; i < n_plot; i++) {
    y_pred_plus_std.at(i) = y_pred_plus_std.at(i) + std_pred.getVal(i, 0);
    y_pred_minus_std.at(i) = y_pred_minus_std.at(i) - std_pred.getVal(i, 0);
    std_pred_vec.at(i) = std_pred_vec.at(i) - 2.0;
    CMatrix x_temp(1, x_dim, x_plot.getVals() + i * x_dim);
    acq_func_plot.setVal(expected_improvement(x_temp, *(BO.gp), BO.y_best, BO.exploration_bias), i);
  }
  // rescale for visualization
  acq_func_plot -= acq_func_plot.min();
  if (acq_func_plot.max() != 0.0) {
    acq_func_plot *= 1.0/acq_func_plot.max();
  }
  acq_func_plot -= 2;
  std::vector<double> acq_func_plot_vec(acq_func_plot.getVals(), acq_func_plot.getVals() + n_plot);
  plot(x_vec, y_true_vec, "k--",
       x_samples_vec, y_samples_vec, "c*",
       x_new_sample_vec, y_new_sample_vec, "rx",
       x_vec, acq_func_plot_vec, "r",
       x_vec, y_pred_vec, "g-",
       x_vec, y_pred_plus_std, "b",
       x_vec, y_pred_minus_std, "b",
       x_vec, std_pred_vec, "b"
      );
  auto lgd = legend("True", "Samples", "New sample", "Acquisition", "y_{pred}", "y_{pred} +/- std", "std");
  lgd->location(legend::general_alignment::topleft);
  xlabel("x");
  title("Bayesian Search");
  show();
}

void plot_BO_sample_times(const CMatrix sample_times, const string func_name) {
  // CMatrix(n_runs, n_iterations) sample_times contains update times for each 
  // sample for each search run of function func_name
  CMatrix mean_sample_times = meanCol(sample_times);
  DIMENSIONMATCH(mean_sample_times.getRows() == 1);
  std::vector<double> sample_times_vec(mean_sample_times.getVals(), 
                                       mean_sample_times.getVals() + mean_sample_times.getNumElements());
  plot(sample_times_vec, "b");
  xlabel("Sample iteration");
  ylabel("Sample update runtime (s)");
  title("Sample update times for function " + func_name + 
        " with " + to_string(sample_times.getRows()) + " runs");
  show();
}

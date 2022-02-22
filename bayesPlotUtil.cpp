#include "bayesPlotUtil.h"

void plot_BO_state(const BayesianSearchModel& BO,
                   const CMatrix& x_plot, const CMatrix& y_plot, 
                   const CMatrix& y_pred, const CMatrix& std_pred, 
                   CMatrix* x_sample, CMatrix* y_sample) {
  // plotting
  int x_dim = BO.x_dim;
  int n_plot = x_plot.getRows();

  // TODO remove these variables, likely unnecessary
  CMatrix* x_sample_plot = x_sample;
  CMatrix* y_sample_plot = y_sample;

  double y_min = BO.y_samples->min();
  double y_max = BO.y_samples->max();
  double range = y_max - y_min;

  std::vector<double> y_true_vec(y_plot.getVals(), y_plot.getVals() + n_plot);
  std::vector<double> y_samples_vec(BO.y_samples->getVals(), BO.y_samples->getVals() + BO.num_samples);
  std::vector<double> y_pred_vec(y_pred.getVals(), y_pred.getVals() + n_plot);
  std::vector<double> std_pred_vec(std_pred.getVals(), std_pred.getVals() + n_plot);
  std::vector<double> y_new_sample_vec(y_sample_plot->getVals(), y_sample_plot->getVals() + 1);

  // 1D plotting
  if (x_dim == 1) {
    assert(x_plot.getCols()==1);

    vector<double> x_vec(x_plot.getVals(), x_plot.getVals() + n_plot);
    std::vector<double> x_samples_vec(BO.x_samples->getVals(), BO.x_samples->getVals() + BO.num_samples);
    std::vector<double> x_new_sample_vec(x_sample_plot->getVals(), x_sample_plot->getVals() + 1);

    std::vector<double> y_pred_plus_std(y_pred_vec);
    std::vector<double> y_pred_minus_std(y_pred_vec);

    CMatrix acq_func_plot(n_plot, 1);
    for (int i = 0; i < n_plot; i++) {
      CMatrix x_temp(1, x_dim, x_plot.getVals() + i * x_dim);
      acq_func_plot.setVal(expected_improvement(x_temp, *(BO.gp), BO.y_best, BO.exploration_bias), i);
    }

    // rescale for visualization
    cout << "range: " << range << endl << " y_min: " << y_min << endl << " y_max: " << y_max << endl;
    acq_func_plot -= acq_func_plot.min();
    if (acq_func_plot.max() != 0.0) {
      acq_func_plot *= 0.15 * range/acq_func_plot.max();
    }
    // place acquisition function below GP plot, taking up 20% of the plotting range
    acq_func_plot += y_min - 0.2 * range;
    std::vector<double> acq_func_plot_vec(acq_func_plot.getVals(), acq_func_plot.getVals() + n_plot);

    for (int i = 0; i < n_plot; i++) {
      y_pred_plus_std.at(i) = y_pred_plus_std.at(i) + std_pred.getVal(i, 0);
      y_pred_minus_std.at(i) = y_pred_minus_std.at(i) - std_pred.getVal(i, 0);
      std_pred_vec.at(i) = std_pred_vec.at(i) + y_min - 0.2 * range;
    }

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
  // 2D plotting
  // produces contour plots for true function, predicted mean, predicted standard deviation, and acquisition function
  // contour plots contain scatter plot of sample points
  else if (x_dim == 2) {
    assert(x_plot.getCols()==2);
    // assume for now that CMatrix arguments have n_plot = n_plot_1d^2 rows of plot values in column major order
    int n_plot_1d = (int)std::sqrt((double)n_plot);
    // auto [X, Y] = meshgrid();

    // hacky and inefficient way to take max/min over columns...
    CMatrix x1_plot(n_plot, 1);
    x1_plot.copyColCol(0, x_plot, 0);
    CMatrix x2_plot(n_plot, 1);
    x2_plot.copyColCol(0, x_plot, 1);
    
    CMatrix x1_mat = linspace(x1_plot.min(), x1_plot.max(), n_plot_1d);
    CMatrix x2_mat = linspace(x2_plot.min(), x2_plot.max(), n_plot_1d);
    vector_1d x1_vec(x1_mat.getVals(), x1_mat.getVals() + n_plot_1d);
    vector_1d x2_vec(x2_mat.getVals(), x2_mat.getVals() + n_plot_1d);

    auto [X1_plot, X2_plot] = meshgrid(x1_vec, x2_vec);

    CMatrix x1_samples(BO.num_samples, 1);
    x1_samples.copyColCol(0, *(BO.x_samples), 0);
    CMatrix x2_samples(BO.num_samples, 1);
    x2_samples.copyColCol(0, *(BO.x_samples), 1);
    std::vector<double> x1_samples_vec(x1_samples.getVals(), x1_samples.getVals() + BO.num_samples);
    std::vector<double> x2_samples_vec(x2_samples.getVals(), x2_samples.getVals() + BO.num_samples);

    // convert CMatrix to 2D vector (column-major order to row-major order)
    vector_2d y_plot_vec;
    vector_2d y_pred_vec;
    vector_2d std_pred_vec;
    // for comparing with plots of test functions
    // bool negate_y_plot = false;
    for (int i = 0; i < n_plot_1d; i++) {
      y_plot_vec.push_back(vector_1d());
      y_pred_vec.push_back(vector_1d());
      std_pred_vec.push_back(vector_1d());
      for (int j = 0; j < n_plot_1d; j++) {
        // double temp = y_plot.getVal(i + j * n_plot_1d);
        // if (negate_y_plot) { temp *= -1; }
        y_plot_vec[i].push_back(y_plot.getVal(i + j * n_plot_1d));
        y_pred_vec[i].push_back(y_pred.getVal(i + j * n_plot_1d));
        std_pred_vec[i].push_back(std_pred.getVal(i + j * n_plot_1d));
      }
    }
    
    int markersize = 2;
    // surface plot of test function
    // figure();
    // surf(X1_plot, X2_plot, y_plot_vec);
    // title("True function");

    // issue in which contourf (for only some test functions) floors larger values to ~1e250 while smaller values 
    // seem floored to 0. Behavior is consistent across runs. surf() plots do not show this behavior. 
    // it might be some kind of allocation issue where values are being deallocated, overwritten, and then later accessed
    // however, seems like an internal error with matplotplusplus given values in y_plot_vec are correct as shown in surf()
    // and by direct examination. Removing surf() does not resolve the issue.
    // I'm leaving this plot given it does work for most of the 2D test functions.
    // other contourf plots are now messed up for mean, std, and acquisition on some runs but not others...
    // tried moving show() inside in variables going out of scope is causing the issue inside matplotplusplus
    // issue still observed
    figure();
    contourf(X1_plot, X2_plot, y_plot_vec);
    hold(on);
    plot(x1_samples_vec, x2_samples_vec, "k.")->marker_size(markersize);
    hold(off);
    title("True function");

    // predictive mean and std
    figure();
    contourf(X1_plot, X2_plot, y_pred_vec);
    hold(on);
    plot(x1_samples_vec, x2_samples_vec, "k.")->marker_size(markersize);
    hold(off);
    title("Predictive mean");

    figure();
    contourf(X1_plot, X2_plot, std_pred_vec);
    hold(on);
    plot(x1_samples_vec, x2_samples_vec, "k.")->marker_size(markersize);
    hold(off);
    title("Predictive STD");

    // currently causing segfaults?
    // acquisition function
    // vector_2d acq_func_plot_vec = 
    //   transform(X1_plot, X2_plot, 
    //     [&x_dim, &BO](double x, double y) {
    //       CMatrix x_temp(1, x_dim);
    //       x_temp.setVal(x, 0);
    //       x_temp.setVal(y, 1);
    //       return expected_improvement(x_temp, *(BO.gp), BO.y_best, BO.exploration_bias);
    //     } );

    // figure();
    // contourf(X1_plot, X2_plot, acq_func_plot_vec);
    // hold(on);
    // plot(x1_samples_vec, x2_samples_vec, "k.")->marker_size(markersize);
    // hold(off);
    // title("Acquisition function");
    show();
  }
  else {
    throw std::invalid_argument("Only x_dim = 1 or 2 supported.");
  }
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

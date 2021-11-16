#include <string>
#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"
#include "cnpy.h"
#include "sklearn_util.h"
#include "CNdlInterfaces.h"
#include <stdexcept>
#include "CBayesianSearch.h"
#include <cmath>
#include <random>
#include <cassert>
#include <matplot/matplot.h>
using namespace matplot;

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>


void plot_BO_state(const BayesianSearchModel& BO, const CMatrix& x_plot, const CMatrix& y_plot, 
                   const CMatrix& y_pred, const CMatrix& std_pred, 
                   CMatrix* x_sample, CMatrix* y_sample);
int testBayesianSearch(string kernel, 
                       string test_func_str,
                       int n_runs=25,
                       int n_iters=250,
                       int n_init_samples=10,
                       double noise_level=0.3,
                       double exp_bias_ratio=0.1,
                       int verbosity=0,
                       bool plotting=false);

class TestFunction {
  public:
    string name;
    int seed = 1235;
    default_random_engine e;
    int x_dim;
    CMatrix x_interval;
    CMatrix y_interval;
    double noise_level = 0.3;
    double y_sol;

    // noise distribution
    normal_distribution<double> dist;

    TestFunction(string func_name, int seed_temp, double noise_level_temp) {
      seed = seed_temp;
      noise_level = noise_level_temp;
      name = func_name;
      default_random_engine e(seed);
      _init();
      dist = normal_distribution(0.0, noise_level * (y_interval(1) - y_interval(0)));
    }

    void _init() {
      // x_dim, x_interval, y_interval set separately for each test function
      if (name.compare("sin") == 0) {
        // gives two global maxima (peaks with equal heights)
        x_dim = 1;
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        x_interval(0, 0) = 3;
        x_interval(0, 1) = 16;
        y_interval(0) = -1;
        y_interval(1) = 1;
      }
      else if (name.compare("mix_gaus") == 0) {
        // mixture of 10 Gaussians
        x_dim = 1;
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        x_interval(0, 0) = 3;
        x_interval(0, 1) = 16;
        y_interval(0) = -1;
        y_interval(1) = 1;
      }
      else {
          throw std::invalid_argument("Test function " + name + " not implemented. Current options: 'sin'");
      }
      assert(x_dim > 0);
      assert(y_interval(1)>=y_interval(0));
      assert(x_interval(1)>=x_interval(0));
      y_sol = y_interval(1);
    }
    
    CMatrix func(const CMatrix& x, bool add_noise=true) {
      CMatrix y(x.getRows(), 1);

      if (name.compare("sin") == 0) {
        assert(x.getCols()==1);
        for (int i = 0; i < x.getRows(); i++) {
          y(i, 0) = sin(x.getVal(i, 0));
        }
      }
      if (add_noise) {
        for (int i = 0; i < x.getRows(); i++) {
          y(i, 0) = y(i, 0) + dist(e);
        }
      }
      return y;
    }

    double solution_error(const CMatrix& x_best) {
      // solution error is relative error of the objective value at the search solution
      // normalized by the output range
      double error = (y_sol - func(x_best, false)(0, 0))/(y_interval(1) - y_interval(0));
      return error;
    }
};


class CClgptest : public CClctrl 
{
 public:
  CClgptest(int arc, char** arv) : CClctrl(arc, arv){}
  void helpInfo(){}
  void helpHeader(){}
};

int main(int argc, char* argv[])
{
  CClgptest command(argc, argv);
  int fail = 0;
  try
  {
    string kern_arg = "matern32";
    string test_func_arg = "sin";
    int n_runs=25;
    int n_iters=250;
    int n_init_samples=10;
    double noise_level=0.3;
    double exp_bias_ratio=0.1;
    int verbosity=3;
    bool plotting=false;

    // left in place if needed for future test arguments
    if (argc > 1) {
      command.setFlags(true);
      while (command.isFlags()) {
        string arg = command.getCurrentArgument();
        if (command.isCurrentArg("-k", "--kernel")) {
          command.incrementArgument();
          kern_arg = command.getCurrentArgument();
        }
        if (command.isCurrentArg("-f", "--func")) {
          command.incrementArgument();
          test_func_arg = command.getCurrentArgument();
        }
        if (command.isCurrentArg("-r", "--n_runs")) {
          command.incrementArgument();
          n_runs = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-i", "--n_iters")) {
          command.incrementArgument();
          n_iters = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-t", "--init_samples")) {
          command.incrementArgument();
          n_init_samples = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-n", "--noise_level")) {
          command.incrementArgument();
          noise_level = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-e", "--exp_bias")) {
          command.incrementArgument();
          exp_bias_ratio = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-v", "--verbosity")) {
          command.incrementArgument();
          verbosity = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-p", "--plot")) {
          plotting = true;
        }
        command.incrementArgument();
      }
    }
    fail += testBayesianSearch(kern_arg, 
                               test_func_arg,
                               n_runs,
                               n_iters,
                               n_init_samples,
                               noise_level,
                               exp_bias_ratio,
                               verbosity,
                               plotting);

    cout << "Number of failures: " << fail << "." << endl;
    command.exitNormal();
  }
  catch(ndlexceptions::Error& err) 
  {
   command.exitError(err.getMessage());
  }
  catch(std::bad_alloc&) 
  {
    command.exitError("Out of memory.");
  }
  catch(std::exception& err) 
  {
    std::string what(err.what());
    command.exitError("Unhandled exception: " + what);
  }

}

int testBayesianSearch(string kernel, 
                       string test_func_str,
                       int n_runs,
                       int n_iters,
                       int n_init_samples,
                       double noise_level,
                       double exp_bias_ratio,
                       int verbosity,
                       bool plotting
)
{
  int fail = 0;
  // string fileName = "np_files" + ndlstrutil::dirSep() + "testSklearn_gpr_" + kernel + ".npz";
  // cnpy::npz_t npz_dict = cnpy::npz_load(fileName.c_str());
  // double* temp = npz_dict["X"].data<double>();
  // CMatrix X(temp, npz_dict["X"].shape[0], npz_dict["X"].shape[1]);

  // // relative difference tolerance
  // double tol = 1e-4;
  // // absolute different tolerance
  // double abs_tol = 1e-5;

  // CMatrix y(npz_dict["y"].data<double>(), npz_dict["y"].shape[0], npz_dict["y"].shape[1]);

  // TODO test integration of Bayesian search, all functions run
  // plot results for debugging

  int n_plot = 200;

  int seed = 1234;

  TestFunction test(test_func_str, seed, noise_level);
  double exp_bias = exp_bias_ratio * (test.y_interval(1) - test.y_interval(0));
  int x_dim = test.x_dim;

  CMatrix search_rel_errors(n_runs, 1);
  // max error of search solution compared to global optimum for an individual run for a run to pass 
  double max_pass_error = 0.05;
  // min proportion of searches that pass
  double min_pass_prob = 0.9;
  // failed runs
  int failures = 0;
  CMatrix sample_times(n_runs, n_iters);
  CMatrix run_times(n_runs, 1);

  for (int run = 0; run < n_runs; run++) {
    seed++;

    CCmpndKern kern(x_dim);
    CMatern32Kern* k = new CMatern32Kern(x_dim);
    kern.addKern(k);
    CWhiteKern* whitek = new CWhiteKern(x_dim);
    kern.addKern(whitek);
    // getSklearnKernels(&kern, npz_dict, &X, true);

    BayesianSearchModel BO(kern, &test.x_interval, exp_bias, n_init_samples, seed, verbosity);

    CMatrix x;
    CMatrix y;
    if (x_dim == 1) {
      x = linspace(test.x_interval(0, 0), test.x_interval(0, 1), n_plot);
      y = test.func(x, false);
    }
    CMatrix y_pred(n_plot, 1);
    CMatrix std_pred(n_plot, 1);

    for (int i = 0; i<n_iters; i++) {
      CMatrix* x_sample = BO.get_next_sample();
      CMatrix* y_sample = new CMatrix(test.func(*x_sample));
      BO.add_sample(*x_sample, *y_sample);

      if (plotting && (verbosity >= 1) && (i > n_init_samples)) {
        BO.get_next_sample();
        if (x_dim == 1) {
          BO.gp->out(y_pred, std_pred, x);
          plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
        }
      }
    }

    // needed for now with janky way I'm deleting CGP gp and CNoise attributes to allow for updating with new 
    // samples
    CMatrix* x_sample = BO.get_next_sample();
    CMatrix* y_sample = new CMatrix(test.func(*x_sample));

    // metrics
    CMatrix* x_best = BO.get_best_solution();
    search_rel_errors(run) = test.solution_error(*x_best);
    cout << "Relative error for run " << run << ": " << search_rel_errors(run) << endl;
    if (search_rel_errors(run) < max_pass_error) {
      failures += 1;
    }

    // plotting
    if (plotting && x_dim == 1) {
      BO.gp->out(y_pred, std_pred, x);
      plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
    }
  }

  double pass_prob = ((double)failures)/((double)n_runs);
  cout << "Proportion of runs passed for " << test_func_str << ": " << pass_prob << endl;
  if (pass_prob < min_pass_prob) {
    fail += 1;
  }

  // TODO test acquisition functions in separate file
  // TODO separate out acquisition functions with templates, virtual function implementation
  
  // TODO test optimization of acquisition function

  return fail;
}

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
  legend("True", "Samples", "New sample", "Acquisition", "y_{pred}", "y_{pred} +/- std", "std");
  xlabel("x");
  title("Bayesian Search");
  show();
}
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
#include <matplot/matplot.h>
using namespace matplot;

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>


int testBayesianSearch(string ratquad);
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
    // left in place if needed for future test arguments
    if (argc > 1) {
      command.setFlags(true);
      while (command.isFlags()) {
        string arg = command.getCurrentArgument();
        if (command.isCurrentArg("-k", "--kernel")) {
          command.incrementArgument();
          arg = command.getCurrentArgument();
          fail += testBayesianSearch(arg);
        }
        command.incrementArgument();
      }
    }
    else {
      fail += testBayesianSearch("matern32");
    }

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

int testBayesianSearch(string kernel)
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
  int init_samples = 10;
  int n_iters = init_samples + 240;
  int x_dim = 1;
  double noise_level = 0.3;
  double x_interval[2] = {-3.0, 16.0};
  double y_interval[2] = {-0.5, 1.0};
  int seed = 1235;
  int verbosity = 0;

  CCmpndKern kern(x_dim);
  CMatern32Kern* k = new CMatern32Kern(x_dim);
  kern.addKern(k);
  CWhiteKern* whitek = new CWhiteKern(x_dim);
  kern.addKern(whitek);
  // getSklearnKernels(&kern, npz_dict, &X, true);

  CMatrix* x_bounds = new CMatrix(x_dim, 2);
  x_bounds->setVal(x_interval[0], 0);
  x_bounds->setVal(x_interval[1], 1);

  double exp_bias = 0.1 * (y_interval[1] - y_interval[0]);
  BayesianSearchModel BO(kern, x_bounds, exp_bias, init_samples, seed, verbosity);

  std::vector<double> x_vec = linspace(x_interval[0], x_interval[1], n_plot);
  CMatrix x(&(x_vec[0]), n_plot, x_dim);
  CMatrix y(n_plot, 1);
  CMatrix y_pred(n_plot, 1);
  CMatrix std_pred(n_plot, 1);

  // test functions
  // assume 1D for now
  auto test_func = [](auto x) { return sin(x); };
  std::vector<double> y_true_vec = transform(x_vec, test_func);

  // noise distribution
  boost::mt19937 rng(seed);
  boost::normal_distribution<> dist(0.0, noise_level*(y_interval[1] - y_interval[0]));
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > norm(rng, dist);

  for (int i = 0; i<n_iters; i++) {
    CMatrix* x_sample = BO.get_next_sample();
    // assumes 1D for now
    CMatrix* y_sample = new CMatrix(test_func(*(x_sample->getVals())) + norm());
    cout << "Sample " << i << ": (x, y): (" << x_sample->getVal(0) << ", " << y_sample->getVal(0) << ")" << endl;
    BO.add_sample(*x_sample, *y_sample);

    if ((i > init_samples) & false) {
      CMatrix* x_sample_plot = x_sample;
      CMatrix* y_sample_plot = y_sample;
      BO.get_next_sample();
      BO.gp->out(y_pred, std_pred, x);

      // plotting
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
        CMatrix x_temp(x.getVals() + i * x_dim, 1, x_dim);
        acq_func_plot.setVal(-expected_improvement(x_temp, *(BO.gp), BO.y_best, BO.exploration_bias), i);
      }
      // rescale for visualization
      acq_func_plot -= acq_func_plot.min();
      if (acq_func_plot.max() != 0.0) {
        acq_func_plot *= 1.0/acq_func_plot.max();
      }
      acq_func_plot -= 2;
      std::vector<double> acq_func_plot_vec(acq_func_plot.getVals(), acq_func_plot.getVals() + n_plot);
      plot(x_vec, y_true_vec, "k.-.",
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
  }
  // needed for now with janky way I'm deleting CGP gp and CNoise attributes to allow for updating with new 
  // samples
  BO.get_next_sample();

  BO.gp->out(y_pred, std_pred, x);

  // plotting
  std::vector<double> x_samples_vec(BO.x_samples->getVals(), BO.x_samples->getVals() + BO.num_samples);
  std::vector<double> y_samples_vec(BO.y_samples->getVals(), BO.y_samples->getVals() + BO.num_samples);
  std::vector<double> y_pred_vec(y_pred.getVals(), y_pred.getVals() + n_plot);
  std::vector<double> y_pred_plus_std(y_pred_vec);
  std::vector<double> y_pred_minus_std(y_pred_vec);
  std::vector<double> std_pred_vec(std_pred.getVals(), std_pred.getVals() + n_plot);
  CMatrix acq_func_plot(n_plot, 1);
  for (int i = 0; i < n_plot; i++) {
    y_pred_plus_std.at(i) = y_pred_plus_std.at(i) + std_pred.getVal(i, 0);
    y_pred_minus_std.at(i) = y_pred_minus_std.at(i) - std_pred.getVal(i, 0);
    std_pred_vec.at(i) -= 2;
    CMatrix x_temp(x.getVals() + i * x_dim, 1, x_dim);
    acq_func_plot.setVal(-expected_improvement(x_temp, *(BO.gp), BO.y_best, BO.exploration_bias), i);
  }
  // rescale for visualization
  acq_func_plot -= acq_func_plot.min();
  if (acq_func_plot.max() != 0.0) {
    acq_func_plot *= 1.0/acq_func_plot.max();
  }
  acq_func_plot -= 2;
  std::vector<double> acq_func_plot_vec(acq_func_plot.getVals(), acq_func_plot.getVals() + n_plot);
  plot(x_vec, y_true_vec, "k",
       x_samples_vec, y_samples_vec, "c*",
       x_vec, acq_func_plot_vec, "r", 
       x_vec, y_pred_vec, "g", 
       x_vec, y_pred_plus_std, "b", 
       x_vec, y_pred_minus_std, "b",
       x_vec, std_pred_vec, "b"
       );
  legend("True", "Samples", "Acquisition", "y_{pred}", "y_{pred} +/- std", "std");
  xlabel("x");
  title("Bayesian Search");
  show();

  // TODO test acquisition functions in separate file
  // TODO separate out acquisition functions with templates, virtual function implementation
  
  // TODO test optimization of acquisition function

  return fail;
}

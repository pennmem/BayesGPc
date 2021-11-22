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
                       int x_dim=1,
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
    int verbosity;

    // noise distribution
    normal_distribution<double> dist;

    TestFunction(string func_name, 
                 int seed_temp, 
                 double noise_level_temp, 
                 int x_dim_temp=1,
                 int verbose=0) {
      seed = seed_temp;
      noise_level = noise_level_temp;
      name = func_name;
      verbosity = verbose;
      x_dim = x_dim_temp;
      default_random_engine e(seed);
      _init();
      dist = normal_distribution(0.0, noise_level * (y_interval(1) - y_interval(0)));
    }

    void _init() {
      // x_dim, x_interval, y_interval set separately for each test function
      if (name.compare("sin") == 0) {
        // gives two global maxima (peaks with equal heights)
        if (x_dim != 1) {
          cout << "Warning: 'sin' test function requires x_dim=1. Overriding user given value." << endl;
        }
        x_dim = 1;
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        x_interval(0, 0) = 3;
        x_interval(0, 1) = 16;
        y_interval(0) = -1;
        y_interval(1) = 1;
      }
      else if (name.compare("quadratic") == 0) {
        // simulates monotonic improvement to edge, testing for edge bias of global optimizer within BO
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        for (int i=0; i<x_dim; i++) {
          x_interval(i, 0) = 0;
          x_interval(i, 1) = 1;
        }
        y_interval(0) = 0;
        y_interval(1) = 1;
      }
      else if (name.compare("quadratic_over_edge") == 0) {
        // simulates monotonic improvement to edge, testing for edge bias of global optimizer within BO
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        for (int i=0; i<x_dim; i++) {
          x_interval(i, 0) = 0;
          x_interval(i, 1) = 1.5;
        }
        y_interval(0) = 0;
        y_interval(1) = 1;
      } 
      // else if (name.compare("mix_gaus") == 0) {
      //   // mixture of 10 Gaussians
      //   x_dim = 1;
      //   x_interval = CMatrix(x_dim, 2);
      //   y_interval = CMatrix(1, 2);
      //   x_interval(0, 0) = 3;
      //   x_interval(0, 1) = 16;
      //   y_interval(0) = -1;
      //   y_interval(1) = 1;
      // }
      else if (name.compare("PS4_1") == 0 || 
               name.compare("PS4_2") == 0 ||
               name.compare("PS4_3") == 0 ||
               name.compare("PS4_4") == 0) {
        // Test functions used in PS4
        if (x_dim != 1) {
          cout << "Warning: 'PS4_X' test functions require x_dim=1. Overriding user given value." << endl;
        }
        x_dim = 1;
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        for (int i=0; i<x_dim; i++) {
          x_interval(i, 0) = 0;
          x_interval(i, 1) = 1;
        }
        // PS4 test functions. Some partial redundancy of functions. 
        // These variations on e.g. simple quadratics provide sanity
        // checks on scale invariance

        // medium-scale quadratic with negative range
        if (name.compare("PS4_1") == 0) {
          y_interval(0) = -0.515918;
          y_interval(1) = -0.154138;
        }
        // multi-modal with different length scales of 
        // underlying functions: clear modes with long-term trend across domain
        else if (name.compare("PS4_2")) {
          y_interval(0) = 0.211798;
          y_interval(1) = 1.4019;
        }
        // small-scale quadratic
        else if (name.compare("PS4_3")) {
          y_interval(0) = -0.000783627;
          y_interval(1) = 0.101074;
        }
        // sin(10x)
        else if (name.compare("PS4_4")) {
          y_interval(0) = -1;
          y_interval(1) = 1;
        }
      }
      // standard optimization test function. Highly multi-modal with
      // one global optimum. Used by Grace Dessert in her BO tests at Nia.
      else if (name.compare("schwefel") == 0) {
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        for (int i=0; i<x_dim; i++) {
          x_interval(i, 0) = -500.0;
          x_interval(i, 1) = 500.0;
        }
        if (x_dim == 1) {
            y_interval(0) = -875;
        }
        else if (x_dim == 2) {
            y_interval(0) = -1800;
        }
        else if (x_dim == 3) {
            y_interval(0) = -1800;
        }
        else if (x_dim == 4) {
            y_interval(0) = -1800;
        }
        else {
            throw std::invalid_argument("Min for test function " + name + " not yet added. " +
                                        "Please run TestFunction.get_func_optimum(true) and hardcode in the minimum value.");
        }
        y_interval(1) = 0;
      }
      // standard multi-modal optimization test function.
      // Used by Grace Dessert in her BO tests at Nia.
      else if (name.compare("hartmann4d") == 0) {
        if (x_dim != 4) {
          cout << "Warning: 'hartmann4d' test function requires x_dim=4. Overriding user given value." << endl;
        }
        x_dim = 4;
        x_interval = CMatrix(x_dim, 2);
        y_interval = CMatrix(1, 2);
        for (int i=0; i<x_dim; i++) {
          x_interval(i, 0) = 0.0;
          x_interval(i, 1) = 1.0;
        }
        y_interval(0) = -1.31105;
        y_interval(1) = 3.51367;
      }
      else {
          throw std::invalid_argument("Test function " + name + " not implemented." + 
                                      " Current options: " + 
                                      "'sin', " + 
                                      "'quadratic', " + 
                                      "'quadratic_over_edge', " + 
                                      "'PS4_1', " + 
                                      "'PS4_2', " + 
                                      "'PS4_3', " + 
                                      "'PS4_4'");
      }
      assert(x_dim > 0);
      assert(y_interval(1)>=y_interval(0));
      assert(x_interval(1)>=x_interval(0));
      y_sol = y_interval(1);
    }
    
    CMatrix func(const CMatrix& x, bool add_noise=true) {
      /* computes a chosen test function for input samples x
      const CMatrix& x (n_samples, x_dim)
      bool add_noise: whether to add uncorrelated Gaussian noise to the samples 
                      with standard deviation noise_level*(y_interval[1]-y_interval[0])
      */

      CMatrix y(x.getRows(), 1);

      if (name.compare("sin") == 0) {
        assert(x.getCols()==1);
        for (int i = 0; i < x.getRows(); i++) {
          y(i, 0) = sin(x.getVal(i, 0));
        }
      }
      else if (name.compare("quadratic") == 0 || name.compare("quadratic_over_edge") == 0) {
        assert(x.getCols()==x_dim);
        for (int i = 0; i < x.getRows(); i++) {
          y(i, 0) = 1.0;
          for (int j = 0; j < x_dim; j++) {
            y(i, 0) -= (x.getVal(i, j) - 1.0)*(x.getVal(i, j) - 1.0);
          }
          y(i, 0) *= 1.0/x_dim;
        }
      }
      // PS4 test functions
      else if (name.compare("PS4_1") == 0) {
        assert(x.getCols()==x_dim);
        for (int i = 0; i < x.getRows(); i++) {
            y(i) = std::exp(-std::pow(x.getVal(i) - 0.8, 2))
                   + std::exp(-std::pow(x.getVal(i) - 0.2, 2))
                   - 0.5 * std::pow(x.getVal(i) - 0.8, 3) - 2.0;
        }
      }
      else if (name.compare("PS4_2") == 0) {
        assert(x.getCols()==x_dim);
        for (int i = 0; i < x.getRows(); i++) {
            y(i) = std::exp(-std::pow(10*x.getVal(i) - 2.0, 2))
                   + std::exp(-std::pow(10*x.getVal(i) - 6.0, 2)/10.0)
                   + 1.0 / (std::pow(10*x.getVal(i), 2) + 1.0);
        }
      }
      else if (name.compare("PS4_3") == 0) {
        assert(x.getCols()==x_dim);
        for (int i = 0; i < x.getRows(); i++) {
            y(i) = 0.2 * (std::exp(-std::pow(x.getVal(i) - 0.8, 2))
                   + std::exp(-std::pow(x.getVal(i) - 0.2, 2))
                   - 0.3 * std::pow(x.getVal(i) - 0.8, 2) - 1.5) + 0.04;
        }
      }
      else if (name.compare("PS4_4") == 0) {
        assert(x.getCols()==x_dim);
        for (int i = 0; i < x.getRows(); i++) {
            y(i) = std::sin(10.0 * x.getVal(i));
        }
      }
      else if (name.compare("schwefel") == 0) {
        assert(x.getCols()==x_dim);
        for (int i = 0; i < x.getRows(); i++) {
          y(i, 0) = 418.9829 * x_dim;
          for (int j = 0; j < x_dim; j++) {
            y(i, 0) -= x.getVal(i, j) * std::sin(std::sqrt(std::abs(x.getVal(i, j))));
          }
          // negate for maximization
          y(i, 0) *= -1;
        }
      }
      else if (name.compare("hartmann4d") == 0) {
        assert(x.getCols()==x_dim);
        CMatrix alpha(1, 4);
        alpha(0) = 1.0; alpha(1) = 1.2; alpha(2) = 3.0; alpha(3) = 3.2;
        double A_array[] = {10,   3,   17,   3.5, 1.7, 8,
                            0.05, 10,  17,   0.1, 8,   14,
                            3,    3.5, 1.7,  10,  17,  8,
                            17,   8,   0.05, 10,  0.1, 14};
        CMatrix A(A_array, 4, 6);
        double P_array[] = {0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886,
                            0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991,
                            0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650,
                            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381 };
        CMatrix P(P_array, 4, 6);
        for (int i = 0; i < x.getRows(); i++) {
            y(i) = 1.1;
            for (int j = 0; j < 4; j++) {
                double inner = 0;
                for (int k = 0; k < 4; k++) {
                    inner -= A(j, k) * std::pow(x.getVal(i, k) - P(j, k) , 2);
                }
                y(i) -= alpha(j) * std::exp(inner);
            }
            // negate for maximization
            y(i) /= -0.839;
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

    struct funcOptimStruct {
      TestFunction* test;
      bool noise;
      bool neg = true;
    };

    // double func_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data);
    double func_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data) {
        funcOptimStruct* optfn_data = reinterpret_cast<funcOptimStruct*>(opt_data);
        CMatrix x_cmat(1, (int)(x.rows()), x.data());
        double out = (optfn_data->test->func(x_cmat, optfn_data->noise))(0);
        if (optfn_data->neg) {out *= -1;}
        return out;
    }

    CMatrix* get_func_optimum(bool get_min = true) {
        CMatrix* x = new CMatrix(1, x_dim);
        Eigen::VectorXd x_optim = Eigen::VectorXd(x_dim);
        Eigen::VectorXd lower_bounds = Eigen::VectorXd(x_dim);
        Eigen::VectorXd upper_bounds = Eigen::VectorXd(x_dim);
        // TODO choose better/random initial values? not strictly needed for global optimization
        for (int i = 0; i < x_dim; i++) {
            x_optim[i] = (x_interval(i, 0) + x_interval(i, 1))/2.0;
            lower_bounds[i] = x_interval(i, 0);
            upper_bounds[i] = x_interval(i, 1);
        }

        CMatrix x_test(1, x_dim, x_optim.data());

        optim::algo_settings_t optim_settings;
        optim_settings.print_level = verbosity;
        if (verbosity >= 1) optim_settings.print_level -= 1;
        optim_settings.vals_bound = true;
        // optim_settings.iter_max = 15;  // doesn't seem to control anything with DE
        optim_settings.rel_objfn_change_tol = 1e-05;
        optim_settings.rel_sol_change_tol = 1e-05;

        optim_settings.de_settings.n_pop = 1000;
        optim_settings.de_settings.n_gen = 1000;

        optim_settings.lower_bounds = lower_bounds;
        optim_settings.upper_bounds = upper_bounds;

        optim_settings.de_settings.initial_lb = lower_bounds;
        optim_settings.de_settings.initial_ub = upper_bounds;

        bool success;

        funcOptimStruct opt_data = {this, false, !get_min};
        success = optim::de(x_optim, 
                            [this](const Eigen::VectorXd& x, 
                                   Eigen::VectorXd* grad_out,
                                   void* opt_data)
                                { return func_optim(x, grad_out, opt_data); },
                            &opt_data,
                            optim_settings);
        if (success) {
            for (int i = 0; i < x_dim; i++) {
                x->setVal(x_optim[i], i);
            }
        }
        else {
            // TODO exceptions aren't printing error messages in Visual Studio
            cout << "Optimization of function " << name << " failed." << endl;
            throw std::runtime_error("Optimization of test function failed.");
        }

        if (verbosity >= 1) {
            CMatrix y = func(*x, false);
            if (get_min) {
                cout << "Min";
            }
            else {
                cout << "Max";
            }
            cout << " found for function " << name <<": (x_best, y_best): (";
            if (x_dim > 1) cout << "[";
            for (int i = 0; i < x_dim; i++) {
                cout  << x->getVal(i) << ", ";
            }
            if (x_dim > 1) cout << "]";
            cout << ", " << y.getVal(0) << ")" << endl;
        }
        return x;
    }
};


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

#include "testBayesianSearch.h"

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
    // default test arguments
    string kern_arg = "matern32";  // not currently used. Kernel is hard coded to matern32 + white
    string test_func_arg = "hartmann4d";
    int n_runs=50;
    int n_iters=250;
    int n_init_samples=25;
    int x_dim = 1;
    double noise_level=0.1;
    double exp_bias_ratio=0.25;
    int verbosity=1;
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
        if (command.isCurrentArg("-d", "--x_dim")) {
          command.incrementArgument();
          x_dim = std::stoi(command.getCurrentArgument());
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
      fail += testBayesianSearch(kern_arg, 
                            test_func_arg,
                            n_runs,
                            n_iters,
                            n_init_samples,
                            x_dim,
                            noise_level,
                            exp_bias_ratio,
                            verbosity,
                            plotting);
    }
    else {
      // fail += testBayesianSearch(kern_arg, 
      //                           "sin",
      //                           n_runs,
      //                           n_iters,
      //                           n_init_samples,
      //                           x_dim,
      //                           noise_level,
      //                           exp_bias_ratio,
      //                           verbosity,
      //                           plotting);
      // fail += testBayesianSearch(kern_arg, 
      //                           "quadratic",
      //                           n_runs,
      //                           n_iters,
      //                           n_init_samples,
      //                           x_dim,
      //                           noise_level,
      //                           exp_bias_ratio,
      //                           verbosity,
      //                           plotting);
      // fail += testBayesianSearch(kern_arg, 
      //                           "quadratic_over_edge",
      //                           n_runs,
      //                           n_iters,
      //                           n_init_samples,
      //                           x_dim,
      //                           noise_level,
      //                           exp_bias_ratio,
      //                           verbosity,
      //                           plotting);
      // fail += testBayesianSearch(kern_arg, 
      //                           "PS4_1",
      //                           n_runs,
      //                           n_iters,
      //                           n_init_samples,
      //                           x_dim,
      //                           noise_level,
      //                           exp_bias_ratio,
      //                           verbosity,
      //                           plotting);
      fail += testBayesianSearch(kern_arg, 
                                "PS4_2",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                x_dim,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
      fail += testBayesianSearch(kern_arg, 
                                "PS4_3",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                x_dim,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
      fail += testBayesianSearch(kern_arg, 
                                "PS4_4",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                x_dim,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
      fail += testBayesianSearch(kern_arg,
                                "schwefel",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                2,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
      fail += testBayesianSearch(kern_arg,
                                "schwefel",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                3,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
      fail += testBayesianSearch(kern_arg,
                                "schwefel",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                4,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
      fail += testBayesianSearch(kern_arg,
                                "hartmann4d",
                                n_runs,
                                n_iters,
                                n_init_samples,
                                x_dim,
                                noise_level,
                                exp_bias_ratio,
                                verbosity,
                                plotting);
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

int testBayesianSearch(string kernel, 
                       string test_func_str,
                       int n_runs,
                       int n_iters,
                       int n_init_samples,
                       int x_dim,
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

  int seed = 1234;

  TestFunction test(test_func_str, seed, noise_level, x_dim, verbosity);

  int n_plot = 400;
  if (test.x_dim == 1 && test.x_interval(1) - test.x_interval(0) > 100.0) {
    n_plot = (int)(test.x_interval(1) - test.x_interval(0));
  }

  // for getting estimates of unknown function optima
  test.verbosity = 1;
  test.get_func_optimum(true);
  test.get_func_optimum(false);
  test.verbosity = verbosity;

  double exp_bias = exp_bias_ratio;
  // kind of cheating, won't know the observation noise ahead of time, but will know to some precision
  double obsNoise = 0.5 * noise_level;

  x_dim = test.x_dim;

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
    cout << "Run " << run << endl;
    seed++;

    CCmpndKern kern(x_dim);
    CMatern32Kern* k = new CMatern32Kern(x_dim);
    kern.addKern(k);
    CWhiteKern* whitek = new CWhiteKern(x_dim);
    kern.addKern(whitek);
    // getSklearnKernels(&kern, npz_dict, &X, true);

    BayesianSearchModel BO(kern, &test.x_interval, obsNoise, exp_bias, n_init_samples, seed, verbosity);

    CMatrix x;
    CMatrix y;
    if (x_dim == 1) {
      double x_interval_len = test.x_interval(0, 1) - test.x_interval(0, 0);
      x = linspace(test.x_interval(0, 0) - 0.0 * x_interval_len, 
                   test.x_interval(0, 1) + 0.0 * x_interval_len, n_plot);
      y = test.func(x, false);
    }
    CMatrix y_pred(n_plot, 1);
    CMatrix std_pred(n_plot, 1);

    for (int i = 0; i<n_iters; i++) {
      CMatrix* x_sample = BO.get_next_sample();
      CMatrix* y_sample = new CMatrix(test.func(*x_sample));
      BO.add_sample(*x_sample, *y_sample);

      if (plotting && (verbosity >= 2) && (i > n_init_samples)) {
        BO.get_next_sample();
        if (x_dim == 1) {
          BO.gp->out(y_pred, std_pred, x);
          plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
        }
      }
    }

    // FIXME needed for now with janky way I'm deleting CGP gp and CNoise attributes to allow for updating with new 
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
  double mean_rel_error_runs = meanCol(search_rel_errors).getVal(0);
  cout << "Proportion of runs passed for " << test_func_str << " with x_dim " << x_dim << ": " << pass_prob << endl;
  cout << "Mean relative error for " << n_runs << " runs: " << mean_rel_error_runs << endl << endl;
  if (pass_prob < min_pass_prob) {
    fail += 1;
  }

  return fail;
}

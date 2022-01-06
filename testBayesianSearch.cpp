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
  CML::EventLog log;
  try
  {
    // default test arguments
    string tag_arg        = "test";
    string results_dir = "results";
    // white kernel currently added to user-specified kernel to obtain full kernel
    // Current options: Matern32, Matern52, RBF, RationalQuadratic, DotProduct, lin (linear kernel)
    string kern_arg       = "Matern32";
    string test_func_arg  = "all";
    int n_runs            = 25;
    int n_iters           = 250;
    int n_init_samples    = 25;
    int x_dim             = 1;
    double noise_level    = 0.1;
    double exp_bias_ratio = 0.25;
    int verbosity         = 1;
    bool full_time_test   = false;
    bool plotting         = false;
    bool debug            = false;

    // left in place if needed for future test arguments
    if (argc > 1) {
      command.setFlags(true);
      while (command.isFlags()) {
        string arg = command.getCurrentArgument();
        if (command.isCurrentArg("-tag", "--tag")) {
          command.incrementArgument();
          tag_arg = command.getCurrentArgument();
        }
        if (command.isCurrentArg("-l", "--logdir")) {
          command.incrementArgument();
          results_dir = command.getCurrentArgument();
        }
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
        if (command.isCurrentArg("-s", "--init_samples")) {
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
        // whether to plot timing results (and to not plot BO state) if plotting is on
        if (command.isCurrentArg("-t", "--timed")) {
          full_time_test = true;
        }
        if (command.isCurrentArg("-v", "--verbosity")) {
          command.incrementArgument();
          verbosity = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-p", "--plot")) {
          plotting = true;
        }
        if (command.isCurrentArg("-debug", "--debug")) {
          debug = true;
        }
        command.incrementArgument();
      }
    }

    string log_dir = results_dir + std::filesystem::path::preferred_separator + tag_arg;
    log_dir += "-func_" + test_func_arg;
    log_dir += "-dim_" + to_string(x_dim);
    log_dir += "-kern_" + kern_arg;
    log_dir += "-runs_" + to_string(n_runs);
    log_dir += "-iters_" + to_string(n_iters);
    log_dir += "-init_samp_" + to_string(n_init_samples);
    log_dir += "-noise_" + to_string(noise_level);
    log_dir += "-exp_bias_" + to_string(exp_bias_ratio);
    log_dir += "_" + getDateTime();

    if (PathExists(log_dir)) { throw std::runtime_error("Log directory " + log_dir + " already exists. Aborting test."); }
    if (std::filesystem::create_directory(log_dir)) { cout << "Made log directory: " << log_dir << endl; }
    else { throw std::runtime_error("Failed to make log directory. Aborting test."); }

    CML::EventLog log;
    log.StartFile_Handler(log_dir + std::filesystem::path::preferred_separator + "log.out");
    log.Log_Handler("Made log file: " + log.get_StartFile() + "\n");
    // separately log full console output
    CML::EventLog full_log;
    if (!debug) {
      full_log.set_log_console_output(true);
      full_log.StartFile_Handler(log_dir + std::filesystem::path::preferred_separator + "full_console_log.out");
    }
    log.Log_Handler(string("git branch:\t") + string(GIT_BRANCH) + string("\n"));
    log.Log_Handler(string("git commit:\t") + string(GIT_COMMIT) + string("\n"));
    log.Log_Handler(string("git URL:\t") + string(GIT_URL) + string("\n"));

    log.Log_Handler("\n");
    log.Flush_Handler();

    // include json logging for parsing
    // keep old .txt logging method for now...
    json config;
    config["func"] = test_func_arg;
    config["dim"] = x_dim;
    config["kern"] = kern_arg;
    config["runs"] = n_runs;
    config["iters"] = n_iters;
    config["init_samp"] = n_init_samples;
    config["noise"] = noise_level;
    config["exp_bias"] = exp_bias_ratio;
    config["datetime"] = getDateTime();
    config["GIT_BRANCH"] = string(GIT_BRANCH);
    config["GIT_COMMIT"] = string(GIT_COMMIT);
    config["GIT_URL"] = string(GIT_URL);
    ofstream config_file(log_dir + std::filesystem::path::preferred_separator + "config.json");
    config_file << config.dump(4);
    config_file.flush();

    json json_log;

    if (test_func_arg.compare("all") == 0) {
      vector<std::pair<std::string, int>> funcs{
                          //  {"sin", 1},
                          //  {"quadratic", 1},
                          //  {"quadratic_over_edge", 1},
                          //  {"PS4_1", 1},
                          //  {"PS4_2", 1},
                          //  {"PS4_3", 1},
                          //  {"PS4_4", 1},
                           {"schwefel", 1},
                           {"schwefel", 2},
                           {"schwefel", 4},
                           {"hartmann4d", 4},
                           {"ackley", 1},
                           {"ackley", 2},
                           {"ackley", 4},
                           {"rastrigin", 1},
                           {"rastrigin", 2},
                           {"rastrigin", 4},
                           {"eggholder", 2},
                           {"sum_squares", 1},
                           {"sum_squares", 2},
                           {"sum_squares", 4},
                           {"rosenbrock", 1},
                           {"rosenbrock", 2},
                           {"rosenbrock", 4},
                          };
      for (auto p : funcs) {
        fail += testBayesianSearch(log,
                        json_log,
                        kern_arg, 
                        p.first,
                        n_runs,
                        n_iters,
                        n_init_samples,
                        p.second,
                        noise_level,
                        exp_bias_ratio,
                        verbosity,
                        full_time_test,
                        plotting);
      }
    }
    else {
      fail += testBayesianSearch(log,
                      json_log,
                      kern_arg, 
                      test_func_arg,
                      n_runs,
                      n_iters,
                      n_init_samples,
                      x_dim,
                      noise_level,
                      exp_bias_ratio,
                      verbosity,
                      full_time_test,
                      plotting);
    }

    log.Log_Handler("Number of failures: " + to_string(fail) + ".");
    log.CloseFile_Handler();

    ofstream json_out(log_dir + std::filesystem::path::preferred_separator + "log.json");
    json_out << json_log.dump(4);
    json_out.flush();

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

int testBayesianSearch(CML::EventLog& log,
                       json& json_log,
                       string kernel, 
                       string test_func_str,
                       int n_runs,
                       int n_iters,
                       int n_init_samples,
                       int x_dim,
                       double noise_level,
                       double exp_bias_ratio,
                       int verbosity,
                       bool full_time_test,
                       bool plotting
)
{
  assert(n_init_samples < n_iters);
  int fail = 0;
  int seed = 1234;

  log.Log_Handler("Test function:\t" + test_func_str + "\n");
  log.Log_Handler("x_dim:\t" + to_string(x_dim) + "\n");
  log.Log_Handler("Kernel:\t" + kernel + "\n");
  log.Log_Handler("n_runs:\t" + to_string(n_runs) + "\n");
  log.Log_Handler("n_iters:\t" + to_string(n_iters) + "\n");
  log.Log_Handler("n_init_samples:\t" + to_string(n_init_samples) + "\n");
  log.Log_Handler("Noise level:\t" + to_string(noise_level) + "\n");
  log.Log_Handler("exp_bias_ratio:\t" + to_string(exp_bias_ratio) + "\n");

  log.Log_Handler("Initial RNG seed:\t" + to_string(seed) + "\n");

  log.Log_Handler("\n");


  TestFunction test(test_func_str, seed, noise_level, x_dim, verbosity);

  int n_plot = 400;
  if (test.x_dim == 1 && test.x_interval(1) - test.x_interval(0) > 400.0) {
    n_plot = (int)(test.x_interval(1) - test.x_interval(0));
  }
  if (test.x_dim == 2) {
    n_plot = 75;
  }
  CMatrix x;
  CMatrix y;
  if (x_dim <= 2) {
    double x_interval_len = test.x_interval(0, 1) - test.x_interval(0, 0);
    if (x_dim == 1) {
      x = linspace(test.x_interval(0, 0) - 0.0 * x_interval_len, 
                   test.x_interval(0, 1) + 0.0 * x_interval_len, n_plot);
    }
    else {
      x = mesh1d(linspace(test.x_interval(0, 0), test.x_interval(0, 1), n_plot),
                 linspace(test.x_interval(1, 0), test.x_interval(1, 1), n_plot));
      // using function name "mesh" instead of "mesh1d" doesn't find function symbol??? 
      // get compiler error with unknown symbol, but name change gives successful compilation?
    }
    y = test.func(x, false);
  }

  CMatrix y_pred(x.getRows(), 1);
  CMatrix std_pred(x.getRows(), 1);

  // for getting estimates of unknown function optima
  // test.verbosity = 1;
  // test.get_func_optimum(true, true);
  // test.get_func_optimum(false, true);
  test.verbosity = verbosity;

  // normalize exploration bias by output range of test function
  // output range unknown for clinical use case but can be bounded and estimated
  double exp_bias = exp_bias_ratio * test.range;

  // assume we know observation noise to some precision; white noise kernel able to adjust for additional noise
  double obsNoise = 0.5 * test.noise_std;

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

  // runs with exceptions
  vector<int> except_runs;

  for (int run = 0; run < n_runs; run++) {
    cout << "Run " << run << endl;
    string func_run = test_func_str + ":run" + to_string(run);
    seed++;
    try {
      cnpy::npz_t npz_dict;
      CKern* k = getSklearnKernel((unsigned int)x_dim, npz_dict, kernel, std::string(""), true);
      CCmpndKern kern(x_dim);
      // CMatern32Kern* k = new CMatern32Kern(x_dim);
      kern.addKern(k);
      CWhiteKern* whitek = new CWhiteKern(x_dim);
      kern.addKern(whitek);
      // getSklearnKernels(&kern, npz_dict, &X, true);

      BayesianSearchModel BO(kern, &test.x_interval, obsNoise * obsNoise, exp_bias, n_init_samples, seed, verbosity);

      clock_t start = clock();
      clock_t sample_update_start;
      for (int i = 0; i<n_iters; i++) {
        sample_update_start = clock();
        CMatrix* x_sample = BO.get_next_sample();
        CMatrix* y_sample = new CMatrix(test.func(*x_sample));
        BO.add_sample(*x_sample, *y_sample);
        sample_times(run, i) = (double)(clock() - sample_update_start)/CLOCKS_PER_SEC;

        if (plotting && (verbosity >= 2) && x_dim <= 2 && (i > n_init_samples)) {
          BO.get_next_sample();
          BO.gp->out(y_pred, std_pred, x);
          plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
        }
      }
      run_times(run) = (double)(clock() - start)/CLOCKS_PER_SEC;

      // FIXME needed for now with janky way I'm deleting CGp gp and CNoise attributes to allow for updating with new 
      // samples
      CMatrix* x_sample = BO.get_next_sample();
      CMatrix* y_sample = new CMatrix(test.func(*x_sample));

      // metrics
      CMatrix* x_best = BO.get_best_solution();
      search_rel_errors(run) = test.solution_error(*x_best);

      cout << "Relative error for run " << run << ": " << search_rel_errors(run) << endl;
      cout << "Run time (s): " << run_times(run) << endl;
      // log.Log_Handler("Relative error for run " + to_string(run) + ": " + to_string(search_rel_errors(run)) + "\n");
      // log.Log_Handler("Run time (s): " + to_string(run_times(run)) + "\n");

      if (search_rel_errors(run) < max_pass_error) {
        failures += 1;
      }

      json_log[func_run]["x"] = to_vector(*(BO.x_samples));
      json_log[func_run]["y"] = to_vector(*(BO.y_samples));
      
      // plotting
      if (plotting && x_dim <= 2 && !full_time_test) {
        BO.gp->out(y_pred, std_pred, x);
        plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
      }

      delete k;
      delete whitek;
    }
    catch(...) { // catch errors for logging/debugging and continue tests
      log.Log_Handler(string("Error in run ") + to_string(run) + string(". Check log."));
      except_runs.push_back(run);
      run_times(run) = nan("");
      search_rel_errors(run) = nan("");
    }
  }

  // performance logging
  double pass_prob = ((double)failures)/((double)n_runs);
  log.Log_Handler("Proportion of runs passed: " + to_string(pass_prob) + "\n");
  json_log[test_func_str]["pass proportion"] = pass_prob;
  log.Log_Handler("Relative error:\n");
  double rel_error_mean = meanCol(search_rel_errors).getVal(0);
  double rel_error_std = stdCol(search_rel_errors).getVal(0);
  double rel_error_sem = rel_error_std/sqrt((double)n_runs);
  log.Log_Handler("Mean +/- STD (SEM):\t" + to_string(rel_error_mean)
                  + " +/- " + to_string(rel_error_std) + " (" + to_string(rel_error_sem) + ")\n");
  json_log[test_func_str]["relative error:mean"] = rel_error_mean;
  json_log[test_func_str]["relative error:std"] = rel_error_std;
  json_log[test_func_str]["relative error:sem"] = rel_error_sem;

  double rel_error_min = search_rel_errors.min();
  log.Log_Handler("Min:\t\t" + to_string(rel_error_min) + "\n");
  json_log[test_func_str]["relative error:min"] = rel_error_min;

  double rel_error_max = search_rel_errors.max();
  log.Log_Handler("Max:\t\t" + to_string(rel_error_max) + "\n");
  json_log[test_func_str]["relative error:max"] = rel_error_max;

  // runtime logging
  double ave_run_time = meanCol(run_times)(0);
  log.Log_Handler("Average run time (s):\t\t" + to_string(ave_run_time) + "\n");
  json_log[test_func_str]["mean run time"] = ave_run_time;
  log.Log_Handler("Average max sample update time (s):\t" 
                  + to_string(meanCol(sample_times)(sample_times.getCols() - 1)) + "\n");
  json_log[test_func_str]["mean max sample update time"] = ave_run_time;
  json_log[test_func_str]["sample times"] = to_vector(sample_times);
  json_log[test_func_str]["run times"] = to_vector(run_times);

  if (except_runs.size() > 0) { 
    log.Log_Handler("Runs with exceptions occurred. See log.json for specific run numbers in item 'exception runs'.\n");
  }
  json_log[test_func_str]["exception runs"] = except_runs;

  log.Log_Handler("\n");
  log.Flush_Handler();

  if (full_time_test && plotting) { plot_BO_sample_times(sample_times, test_func_str); }

  if (pass_prob < min_pass_prob) {
    fail += 1;
  }
  return fail;
}

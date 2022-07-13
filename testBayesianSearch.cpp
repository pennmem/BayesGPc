#include "testBayesianSearch.h"
#ifdef _WIN
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

class CClgptest : public CClctrl 
{
 public:
  CClgptest(int arc, char** arv) : CClctrl(arc, arv){}
  void helpInfo(){}
  void helpHeader(){}
};

int main(int argc, char* argv[])
{
  #ifdef _WIN
  wchar_t wchar[2] = {fs::path::preferred_separator, '\0'};
  std::wstring ws(wchar);
  std::string separator(std::string(ws.begin(), ws.end()));
  #else
  std::string separator(1, fs::path::preferred_separator);
  #endif
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
    int n_grid            = 0;  // use grid search if greater than zero with grid having roughly n_grid total points with equal density across dimensions
    double noise_level    = 0.1;
    double exp_bias_ratio = 0.25;
    double lengthscale_lb = 0.1;
    double lengthscale_ub = 2.0;
    double white_lb       = 0.1;
    double white_ub       = 4.0;
    int seed              = 1234;
    int verbosity         = 0;
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
        if (command.isCurrentArg("-s", "--n_init_samples")) {
          command.incrementArgument();
          n_init_samples = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-d", "--x_dim")) {
          command.incrementArgument();
          x_dim = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-g", "--n_grid")) {
          command.incrementArgument();
          n_grid = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-n", "--noise_level")) {
          command.incrementArgument();
          noise_level = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-lsl", "--lenscale_lb")) {
          command.incrementArgument();
          lengthscale_lb = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-lsu", "--lenscale_ub")) {
          command.incrementArgument();
          lengthscale_ub = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-wl", "--white_lb")) {
          command.incrementArgument();
          white_lb = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-wu", "--white_ub")) {
          command.incrementArgument();
          white_ub = std::stod(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-e", "--exp_bias")) {
          command.incrementArgument();
          exp_bias_ratio = std::stod(command.getCurrentArgument());
        }
        // whether to plot timing results (and to not plot BO state) if plotting is on
        if (command.isCurrentArg("-t", "--timed")) {
          full_time_test = true;
        }
        if (command.isCurrentArg("-seed", "--seed")) {
          command.incrementArgument();
          seed = std::stoi(command.getCurrentArgument());
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

    string log_dir = results_dir + separator + tag_arg;
    log_dir += "-func_" + test_func_arg;
    log_dir += "-dim_" + to_string(x_dim);
    log_dir += "-kern_" + kern_arg;
    log_dir += "-runs_" + to_string(n_runs);
    log_dir += "-iters_" + to_string(n_iters);
    log_dir += "-init_samp_" + to_string(n_init_samples);
    log_dir += "-noise_" + to_string(noise_level);
    log_dir += "-len_lb_" + to_string(lengthscale_lb);
    log_dir += "-len_ub_" + to_string(lengthscale_ub);
    log_dir += "-white_lb_" + to_string(white_lb);
    log_dir += "-white_ub_" + to_string(white_ub);
    log_dir += "-exp_bias_" + to_string(exp_bias_ratio);
    log_dir += "-n_grid_" + to_string(n_grid);
    log_dir += "_" + getDateTime();

    if (PathExists(log_dir)) { throw std::runtime_error("Log directory " + log_dir + " already exists. Aborting test."); }
    string json_dir = log_dir + separator + string("json_logs");
    #ifdef _WIN  // std::filesystem not working with mingw w64 7.3.0
    QString qstr = QString::fromStdString(log_dir);
    QDir qdir = QDir::current();
    if (qdir.mkdir(qstr)) {
      cout << "Made log directory: " << log_dir << endl;
      QString json_qstr = QString::fromStdString(json_dir);
      QDir json_qdir = QDir::current();
      if (qdir.mkdir(json_qstr)) { cout << "Made JSON log directory: " << json_dir << endl; }
      else { throw std::runtime_error("Failed to make JSON log directory. Aborting test."); }
    }
    #else
    if (fs::create_directory(log_dir)) {
      cout << "Made log directory: " << log_dir << endl;
      if (fs::create_directory(json_dir)) {
          cout << "Made log directory: " << log_dir << endl;
      }
      else { throw std::runtime_error("Failed to make JSON log directory. Aborting test."); }
    }
    #endif
    else { throw std::runtime_error("Failed to make log directory. Aborting test."); }

    CML::EventLog log;
    log.StartFile_Handler(log_dir + separator + "log.out");
    log.Log_Handler("Made log file: " + log.get_StartFile() + "\n");
    // separately log full console output
    CML::EventLog full_log;
    if (!debug) {
      full_log.set_log_console_output(true);
      full_log.StartFile_Handler(log_dir + separator + "full_console_log.out");
    }
    log.Log_Handler(string("git branch:\t") + string(GIT_BRANCH) + string("\n"));
    log.Log_Handler(string("git commit:\t") + string(GIT_COMMIT) + string("\n"));
    log.Log_Handler(string("git URL:\t") + string(GIT_URL) + string("\n"));

    log.Log_Handler("\n");
    log.Flush_Handler();

    // include json logging for parsing
    // keep old .txt logging method for now...
    json config;
    config["impl"] = string("CBay");
    config["tag"] = tag_arg;
    config["logdir"] = log_dir;
    config["verbosity"] = verbosity;
    config["plot"] = plotting;
    config["seed"] = seed;

    config["func"] = test_func_arg;
    config["dim"] = x_dim;
    config["n_grid"] = n_grid;
    config["kern"] = kern_arg;
    config["n_runs"] = n_runs;
    config["n_iters"] = n_iters;
    config["n_init_samp"] = n_init_samples;
    config["noise_level"] = noise_level;
    config["lengthscale_lb"] = lengthscale_lb;
    config["lengthscale_ub"] = lengthscale_ub;
    config["white_lb"] = white_lb;
    config["white_ub"] = white_ub;
    config["exp_bias"] = exp_bias_ratio;
    config["datetime"] = getDateTime();
    config["GIT_BRANCH"] = string(GIT_BRANCH);
    config["GIT_COMMIT"] = string(GIT_COMMIT);
    config["GIT_URL"] = string(GIT_URL);
    ofstream config_file(log_dir + separator + "config.json");
    config_file << config.dump(4);
    config_file.flush();

    if (test_func_arg.compare("all") == 0) {
      vector<std::pair<std::string, int>> funcs{
                           {"sin", 1},
                          //  {"null", 1},
                           {"quadratic", 1},
                           {"quadratic_over_edge", 1},
                          //  {"PS4_1", 1},
                           {"PS4_2", 1},
                           {"PS4_3", 1},
                          //  {"PS4_4", 1},
                          //  {"schwefel", 1},
                          //  {"schwefel", 2},
                          //  {"schwefel", 4},
                          //  {"hartmann4d", 4},
                          //  {"ackley", 1},
                          //  {"ackley", 2},
                          //  {"ackley", 4},
                          //  {"rastrigin", 1},
                          //  {"rastrigin", 2},
                          //  {"rastrigin", 4},
                          //  {"eggholder", 2},
                          //  {"sum_squares", 1},
                          //  {"sum_squares", 2},
                          //  {"sum_squares", 4},
                          //  {"rosenbrock", 2},
                          //  {"rosenbrock", 4},
                          };
      for (auto p : funcs) {
        fail += testBayesianSearch(log,
                        json_dir,
                        kern_arg, 
                        p.first,
                        n_runs,
                        n_iters,
                        n_init_samples,
                        p.second,
                        n_grid,
                        noise_level,
                        lengthscale_lb,
                        lengthscale_ub,
                        white_lb,
                        white_ub,
                        exp_bias_ratio,
                        verbosity,
                        full_time_test,
                        plotting,
                        seed);
      }
    }
    else {
      fail += testBayesianSearch(log,
                      json_dir,
                      kern_arg, 
                      test_func_arg,
                      n_runs,
                      n_iters,
                      n_init_samples,
                      x_dim,
                      n_grid,
                      noise_level,
                      lengthscale_lb,
                      lengthscale_ub,
                      white_lb,
                      white_ub,
                      exp_bias_ratio,
                      verbosity,
                      full_time_test,
                      plotting,
                      seed);
    }

    log.Log_Handler("Number of failures: " + to_string(fail) + ".\n");
    log.CloseFile_Handler();

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
                       string& json_dir,
                       string kernel, 
                       string test_func_str,
                       int n_runs,
                       int n_iters,
                       int n_init_samples,
                       int x_dim,
                       int n_grid,
                       double noise_level,
                       double lengthscale_lb,
                       double lengthscale_ub,
                       double white_lb,
                       double white_ub,
                       double exp_bias_ratio,
                       int verbosity,
                       bool full_time_test,
                       bool plotting,
                       int seed
)
{
  json json_log;

  assert(n_init_samples < n_iters);
  int fail = 0;
  #ifdef _WIN
  wchar_t wchar[2] = {fs::path::preferred_separator, '\0'};
  std::wstring ws(wchar);
  std::string separator(std::string(ws.begin(), ws.end()));
  #else
  std::string separator(1, fs::path::preferred_separator);
  #endif

  log.Log_Handler("Test function:\t" + test_func_str + "\n");
  log.Log_Handler("x_dim:\t" + to_string(x_dim) + "\n");
  log.Log_Handler("Kernel:\t" + kernel + "\n");
  log.Log_Handler("n_runs:\t" + to_string(n_runs) + "\n");
  log.Log_Handler("n_iters:\t" + to_string(n_iters) + "\n");
  log.Log_Handler("n_init_samples:\t" + to_string(n_init_samples) + "\n");
  log.Log_Handler("Noise level:\t" + to_string(noise_level) + "\n");
  log.Log_Handler("lengthscale lower bound:\t" + to_string(lengthscale_lb) + "\n");
  log.Log_Handler("lengthscale upper bound:\t" + to_string(lengthscale_ub) + "\n");
  log.Log_Handler("white variance lower bound:\t" + to_string(white_lb) + "\n");
  log.Log_Handler("white variance upper bound:\t" + to_string(white_ub) + "\n");
  log.Log_Handler("exp_bias_ratio:\t" + to_string(exp_bias_ratio) + "\n");

  log.Log_Handler("Initial RNG seed:\t" + to_string(seed) + "\n");

  log.Log_Handler("\n");

  string fd = test_func_str + ":d" + to_string(x_dim);

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

//  // for getting estimates of unknown function optima
//  test.verbosity = 1;
//  #ifdef _WIN
//  std::vector<CMatrix> grid_vals_test;
//  if (x_dim == 1) { n_grid = 100001; }
//  else if (x_dim == 2) { n_grid = 10001; }
//  else if (x_dim == 3) { n_grid = 501; }
//  else if (x_dim == 4) { n_grid = 101; }
//  else { throw std::runtime_error("Need to hard code n_grid for x_dim >= 5"); }
//  for (int i = 0; i < x_dim; i++) {
//      CMatrix grid1D = linspace(test.x_interval.getVal(i,0),
//                                test.x_interval.getVal(i,1),
//                                n_grid);
//      grid_vals_test.push_back(grid1D);
//  }
//  test.grid_vals = grid_vals_test;
//  #endif  // _WIN
//  test.get_func_optimum(true);
//  test.get_func_optimum(false);
//  test.verbosity = verbosity;
//  throw std::runtime_error("stopping for tests");

  // normalize exploration bias by output range of test function
  // output range unknown for clinical use case but can be bounded and estimated
  double exp_bias = exp_bias_ratio * test.range;

  // assume we know observation noise to some precision; white noise kernel able to adjust for additional noise
  // or set to zero for comparison with sklearn/skopt
  double obsNoise = 0.0;  //0.1 * test.noise_std;

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
  // shape: (runs, samples, sample dimension)
  vector<vector<vector<double>>> x_samples_runs;
  vector<vector<vector<double>>> y_samples_runs;
  // shape: (runs, samples, n_kernel_params)
  vector<vector<vector<double>>> sample_search_states(n_runs, vector<vector<double>>());

  TestFunction dummy_test_fcn(test_func_str, seed, noise_level, x_dim, verbosity);
  vector<CMatrix> grid_vals;
  if (n_grid > 0) {
    int n_grid_dim = n_grid;  // for simple allocation without truncation, useful particularly for smaller n_grid values
    // int n_grid_dim = (int)std::pow((double)n_grid, 1.0/((double)x_dim));  // for fixed sample budget
    for (int i = 0; i < x_dim; i++) {
        CMatrix grid1D = linspace(dummy_test_fcn.x_interval.getVal(i,0),
                                  dummy_test_fcn.x_interval.getVal(i,1),
                                  n_grid_dim);
        grid_vals.push_back(grid1D);
    }
    dummy_test_fcn.grid_vals = grid_vals;
    dummy_test_fcn.y_interval(0) = dummy_test_fcn.func(*dummy_test_fcn.get_func_optimum(true), true).getVal(0);
    dummy_test_fcn.y_interval(1) = dummy_test_fcn.func(*dummy_test_fcn.get_func_optimum(false), false).getVal(0);
    dummy_test_fcn.y_sol = dummy_test_fcn.y_interval.getVal(1);
    dummy_test_fcn.range = dummy_test_fcn.y_interval(1) - dummy_test_fcn.y_interval(0);
    assert(dummy_test_fcn.range >= 0);
  }

  for (int run = 0; run < n_runs; run++) {
    cout << "Run " << run << endl;
    seed++;
    // TODO: RDD: fix: currently reseeding test function with reinitialization
    // strange RNG seeding behavior with RNG std::default_random_engine, 
    // possibly a type issue, leads to binary switching of 
    // seed states between samples even after reseeding so that, 
    // after reseeding, identical behavior results only after an even number of samples are drawn
    // may also be driven by the default RNG generator used...
    TestFunction test_run(test_func_str, seed, noise_level, x_dim, verbosity);
    if (n_grid > 0) {
      test_run.y_interval = dummy_test_fcn.y_interval;
      test_run.y_sol = dummy_test_fcn.y_sol;
      test_run.range = dummy_test_fcn.range;
    }

    try {
      CCmpndKern kern = getTestKernel(kernel, test_run, 
                                      lengthscale_lb, lengthscale_ub, 
                                      white_lb, white_ub);

      BayesianSearchModel BO(kern, test_run.x_interval, obsNoise * obsNoise, exp_bias, n_init_samples, seed, verbosity, grid_vals);
      if (n_grid > 0) { BO.init_points_on_grid = true; }
      if (run == 0) {
          json_log[fd]["kernel_structure"] = BO.kern->json_structure();  // kept for backwards compatibility, TODO remove
          json_log[fd]["BO_structure"] = BO.json_structure();
          json_log[fd]["BO_state"] = BO.json_state();  // TODO remove, just here for testing
      }

      clock_t start = clock();
      clock_t sample_update_start;
      for (int i = 0; i < n_iters; i++) {
        sample_update_start = clock();
        CMatrix* x_sample = BO.get_next_sample();
        CMatrix* y_sample = new CMatrix(test_run.func(*x_sample));
        BO.add_sample(*x_sample, *y_sample);
        if (verbosity >= 1) { cout << endl << endl; }

        // logging
        if (i >= n_init_samples) { sample_search_states[run].push_back(BO.gp->pkern->state()); }
        sample_times(run, i) = (double)(clock() - sample_update_start)/CLOCKS_PER_SEC;

        #ifndef _WIN
        if (plotting && (verbosity >= 3) && x_dim <= 2 && (i > n_init_samples)) {
          BO.get_next_sample();
          BO.gp->out(y_pred, std_pred, x);
          // BO.gp->out_sem(y_pred, std_pred, x);
          plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
        }
        #endif
      }
      run_times(run) = (double)(clock() - start)/CLOCKS_PER_SEC;

      // FIXME needed for now with janky way I'm deleting CGp gp and CNoise attributes to allow for updating with new 
      // samples
      cout << "Computing next sample after last sample in run:" << endl;
      CMatrix* x_sample = BO.get_next_sample();
      CMatrix* y_sample = new CMatrix(test_run.func(*x_sample));

      // metrics
      CMatrix* x_best = BO.get_best_solution();
      search_rel_errors(run) = test_run.solution_error(*x_best);

      cout << "Relative error for run " << run << ": " << search_rel_errors(run) << endl;
      cout << "Run time (s): " << run_times(run) << endl;
      // log.Log_Handler("Relative error for run " + to_string(run) + ": " + to_string(search_rel_errors(run)) + "\n");
      // log.Log_Handler("Run time (s): " + to_string(run_times(run)) + "\n");

      if (search_rel_errors(run) < max_pass_error) {
        failures += 1;
      }
      x_samples_runs.push_back(to_vector(*(BO.x_samples)));
      y_samples_runs.push_back(to_vector(*(BO.y_samples)));
      
      // plotting
      #ifndef _WIN
      if (plotting && x_dim <= 2 && !full_time_test) {
        // BO.gp->out(y_pred, std_pred, x);
        BO.gp->out_sem(y_pred, std_pred, x);
        plot_BO_state(BO, x, y, y_pred, std_pred, x_sample, y_sample);
      }
      #endif
    }
    catch(const std::exception& e) { // catch errors for logging/debugging and continue tests
      log.Log_Handler(string("Error in run ") + to_string(run) + string("\n") + string("Check log.\n"));
      cout << "Error message: " << e.what() << endl;
      except_runs.push_back(run);
      run_times(run) = nan("");
      search_rel_errors(run) = nan("");
    }
  }

  // performance and model logging
  json_log[fd]["x_samples"] = x_samples_runs;
  json_log[fd]["y_samples"] = y_samples_runs;
  json_log[fd]["relative errors"] = to_vector(search_rel_errors);
  json_log[fd]["kernel_states"] = sample_search_states; 

  double pass_prob = ((double)failures)/((double)n_runs);
  log.Log_Handler("Proportion of runs passed: " + to_string(pass_prob) + "\n");
  log.Log_Handler("Relative error:\n");
  double rel_error_mean = meanCol(search_rel_errors).getVal(0);
  double rel_error_std = stdCol(search_rel_errors).getVal(0);
  double rel_error_sem = rel_error_std/sqrt((double)n_runs);
  log.Log_Handler("Mean +/- STD (SEM):\t" + to_string(rel_error_mean)
                  + " +/- " + to_string(rel_error_std) + " (" + to_string(rel_error_sem) + ")\n");

  double rel_error_min = search_rel_errors.min();
  log.Log_Handler("Min:\t\t" + to_string(rel_error_min) + "\n");

  double rel_error_max = search_rel_errors.max();
  log.Log_Handler("Max:\t\t" + to_string(rel_error_max) + "\n");

  // runtime logging
  double ave_run_time = meanCol(run_times)(0);
  log.Log_Handler("Average run time (s):\t\t" + to_string(ave_run_time) + "\n");
  log.Log_Handler("Average max sample update time (s):\t" 
                  + to_string(meanCol(sample_times)(sample_times.getCols() - 1)) + "\n");
  json_log[fd]["sample times"] = to_vector(sample_times);
  json_log[fd]["run times"] = to_vector(run_times);

  if (except_runs.size() > 0) { 
    log.Log_Handler("Runs with exceptions occurred. See log.json for specific run numbers in item 'exception runs'.\n");
  }
  json_log[fd]["exception runs"] = except_runs;

  log.Log_Handler("\n");
  log.Flush_Handler();

  ofstream json_out(json_dir + separator + string("log_") + fd + string(".json"));
  json_out << json_log.dump(4);
  json_out.flush();

  #ifndef _WIN
  if (full_time_test && plotting) { plot_BO_sample_times(sample_times, test_func_str); }
  #endif

  if (pass_prob < min_pass_prob) {
    fail += 1;
  }
  return fail;
}

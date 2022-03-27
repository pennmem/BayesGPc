#include "testBayesianSearch.h"
#include "CSearchComparison.h"
#include "cnpy.h"

int testSearchComparison(CML::EventLog& log,
                        json& json_log,
                        string kernel, 
                        string test_func_str,
                        int n_runs=25,
                        int n_iters=250,
                        int n_init_samples=10,
                        int x_dim=1,
                        int n_way=1,
                        double mean_diff=0.1,
                        double noise_level=0.3,
                        double exp_bias_ratio=0.1,
                        int verbosity=0,
                        bool full_time_test=false,
                        bool plotting=false,
                        int seed=1234);

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
    string tag_arg        = "test_CSearchComparison";
    string results_dir = "results/search_comparison";
    // white kernel currently added to user-specified kernel to obtain full kernel
    // Current options: Matern32, Matern52, RBF, RationalQuadratic, DotProduct, lin (linear kernel)
    string kern_arg       = "Matern32";
    string test_func_arg  = "all";
    int n_runs            = 25;
    int n_iters           = 250;
    int n_init_samples    = 25;
    int x_dim             = 1;
    int n_way             = 2;
    double mean_diff      = 0.1;  // constant difference between test function(s)
    double noise_level    = 0.1;
    double exp_bias_ratio = 0.25;
    int seed              = 1234;
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
        if (command.isCurrentArg("-s", "--n_init_samples")) {
          command.incrementArgument();
          n_init_samples = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-d", "--x_dim")) {
          command.incrementArgument();
          x_dim = std::stoi(command.getCurrentArgument());
        }
        if (command.isCurrentArg("-n_way", "--n_way")) {
          command.incrementArgument();
          n_way = std::stoi(command.getCurrentArgument());
          assert(n_way > 1);
        }
        if (command.isCurrentArg("-m", "--mean_diff")) {
          command.incrementArgument();
          mean_diff = std::stod(command.getCurrentArgument());
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

    string log_dir = results_dir + std::filesystem::path::preferred_separator + tag_arg;
    log_dir += "-func_" + test_func_arg;
    log_dir += "-dim_" + to_string(x_dim);
    log_dir += "-kern_" + kern_arg;
    log_dir += "-runs_" + to_string(n_runs);
    log_dir += "-iters_" + to_string(n_iters);
    log_dir += "-init_samp_" + to_string(n_init_samples);
    log_dir += "-n_way_" + to_string(n_way);
    log_dir += "-mean_diff_" + to_string(mean_diff);
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
    config["impl"] = string("CSearchComparison");
    config["tag"] = tag_arg;
    config["logdir"] = log_dir;
    config["verbosity"] = verbosity;
    config["plot"] = plotting;
    config["seed"] = seed;

    config["func"] = test_func_arg;
    config["dim"] = x_dim;
    config["kern"] = kern_arg;
    config["n_runs"] = n_runs;
    config["n_iters"] = n_iters;
    config["n_init_samp"] = n_init_samples;
    config["n_way"] = n_way;
    config["mean_diff"] = mean_diff;
    config["noise_level"] = noise_level;
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
                           {"sin", 1},
                           {"quadratic", 1},
                           {"quadratic_over_edge", 1},
                          //  {"PS4_1", 1},
                          //  {"PS4_2", 1},
                          //  {"PS4_3", 1},
                          //  {"PS4_4", 1},
                           {"schwefel", 1},
                           {"schwefel", 2},
                        //    {"schwefel", 4},
                        //    {"hartmann4d", 4},
                        //    {"ackley", 1},
                        //    {"ackley", 2},
                        //    {"ackley", 4},
                        //    {"rastrigin", 1},
                        //    {"rastrigin", 2},
                        //    {"rastrigin", 4},
                        //    {"eggholder", 2},
                           {"sum_squares", 1},
                           {"sum_squares", 2},
                        //    {"sum_squares", 4},
                        //    {"rosenbrock", 2},
                        //    {"rosenbrock", 4},
                          };
      for (auto p : funcs) {
        fail += testSearchComparison(log,
                        json_log,
                        kern_arg, 
                        p.first,
                        n_runs,
                        n_iters,
                        n_init_samples,
                        p.second,
                        n_way,
                        mean_diff,
                        noise_level,
                        exp_bias_ratio,
                        verbosity,
                        full_time_test,
                        plotting,
                        seed);
      }
    }
    else {
      fail += testSearchComparison(log,
                      json_log,
                      kern_arg, 
                      test_func_arg,
                      n_runs,
                      n_iters,
                      n_init_samples,
                      x_dim,
                      n_way,
                      mean_diff,
                      noise_level,
                      exp_bias_ratio,
                      verbosity,
                      full_time_test,
                      plotting,
                      seed);
    }

    log.Log_Handler("Number of failures: " + to_string(fail) + ".\n");
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

int testSearchComparison(CML::EventLog& log,
                       json& json_log,
                       string kernel, 
                       string test_func_str,
                       int n_runs,
                       int n_iters,
                       int n_init_samples,
                       int x_dim,
                       int n_way,
                       double mean_diff,
                       double noise_level,
                       double exp_bias_ratio,
                       int verbosity,
                       bool full_time_test,
                       bool plotting,
                       int seed
)
{
  assert(n_init_samples < n_iters);
  int fail = 0;
  int correct_model = 0;

  log.Log_Handler("Test function:\t" + test_func_str + "\n");
  log.Log_Handler("x_dim:\t" + to_string(x_dim) + "\n");
  log.Log_Handler("Kernel:\t" + kernel + "\n");
  log.Log_Handler("n_runs:\t" + to_string(n_runs) + "\n");
  log.Log_Handler("n_iters:\t" + to_string(n_iters) + "\n");
  log.Log_Handler("n_init_samples:\t" + to_string(n_init_samples) + "\n");
  log.Log_Handler("n_way:\t" + to_string(n_way) + "\n");
  log.Log_Handler("correct_model:\t" + to_string(correct_model) + "\n");
  log.Log_Handler("Mean difference:\t" + to_string(mean_diff) + "\n");
  log.Log_Handler("Noise level:\t" + to_string(noise_level) + "\n");
  log.Log_Handler("exp_bias_ratio:\t" + to_string(exp_bias_ratio) + "\n");

  log.Log_Handler("Initial RNG seed:\t" + to_string(seed) + "\n");

  log.Log_Handler("\n");

  string fd = test_func_str + ":d" + to_string(x_dim);
  json_log[fd] = json({});
  json_log[fd]["run"] = json::array();

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
  // test.get_func_optimum(true);
  // test.get_func_optimum(false);
  // test.verbosity = verbosity;

  // normalize exploration bias by output range of test function
  // output range unknown for clinical use case but can be bounded and estimated
  double exp_bias = exp_bias_ratio * test.range;

  // assume we know observation noise to some precision; white noise kernel able to adjust for additional noise
  double obsNoise = 0.1 * test.noise_std;

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
  vector<double> correct_inferences;
  vector<double> pvals;
  double alpha = 0.05;

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

    try {
      CCmpndKern kern = getTestKernel(kernel, test_run.x_interval(0, 1) - test_run.x_interval(0, 0), x_dim);
      // TODO change over all function arguments to CKern
      vector<CCmpndKern> kernels;
      vector<CMatrix> param_bounds;
      vector<double> observation_noises;
      vector<double> exploration_biases;
      vector<int> init_samples;
      vector<int> rng_seeds;
      for (int w = 0; w < n_way; w++) {
          kernels.push_back(kern);
          param_bounds.push_back(test_run.x_interval);
          observation_noises.push_back(obsNoise * obsNoise);
          exploration_biases.push_back(exp_bias);
          init_samples.push_back(n_init_samples);
          rng_seeds.push_back(seed + run * n_way * n_iters);
      }

      CSearchComparison search(n_way, alpha, kernels, param_bounds, observation_noises,
              exploration_biases, init_samples, rng_seeds, verbosity);
      // use identical kernels across functions for now
      if (run == 0) { json_log[fd]["kernel_structure"] = kern.json_structure(); }

      clock_t start = clock();
      clock_t sample_update_start;
      json_log[fd]["run"].push_back(json::array());
      cout << json_log << endl;
      for (int w = 0; w < n_way; w++) {
        json_log[fd]["run"][run].push_back(json({}));
        json_log[fd]["run"][run][w]["relative errors"] = json::array();
        json_log[fd]["run"][run][w]["kernel_states"] = json::array();
        json_log[fd]["run"][run][w]["x_samples"] = json::array();
        json_log[fd]["run"][run][w]["y_samples"] = json::array();
        for (int i = 0; i < n_iters; i++) {
          CMatrix* x_sample = search.get_next_sample(w);
          // difference in means between group/model 0 vs. others
          double val = test_run.func(*x_sample).getVal(0) + (w == correct_model ? mean_diff * test_run.noise_std : 0.0);
          CMatrix* y_sample = new CMatrix(val);
          search.add_sample(w, *x_sample, *y_sample);
          cout << endl << endl;

          // logging
          if (i >= n_init_samples) { json_log[fd]["run"][run][w]["kernel_states"].push_back(search.models[w].gp->pkern->state()); }
          json_log[fd]["run"][run][w]["x_samples"].push_back(to_vector(*x_sample));
          json_log[fd]["run"][run][w]["y_samples"].push_back(to_vector(*y_sample));
        }

        // FIXME needed for now with janky way I'm deleting CGp gp and CNoise attributes to allow for updating with new 
        // samples
        cout << "Computing next sample after last sample in run:" << endl;
        CMatrix* x_sample = search.models[w].get_next_sample();
        CMatrix* y_sample = new CMatrix(test_run.func(*x_sample));

        // run metrics
        CMatrix* x_best;
        x_best = search.models[w].get_best_solution();
        // allow extra dimension for generalizing to getting error at each sample
        json_log[fd]["run"][run][w]["relative errors"][0] = test_run.solution_error(*x_best);
        cout << "Relative error for run " << run << ", group " << w << ": " << json_log[fd]["run"][run][w]["relative errors"][0] << endl;

        // plotting
        if (plotting && x_dim <= 2 && !full_time_test) {
          search.models[w].gp->out_sem(y_pred, std_pred, x);
          plot_BO_state(search.models[w], x, y, y_pred, std_pred, x_sample, y_sample);
        }
      }

      // compare groups
      ComparisonStruct sol = search.get_best_solution();
      vector<vector<double>> xs;
      for (int w = 0; w < n_way; w++) {
        xs.push_back(to_vector1D(*(sol.xs[w])));
      }
      // store comparison_structs in json for first group to simplify json structure
      json_log[fd]["run"][run][0]["comparison_struct"] = json({});
      json_log[fd]["run"][run][0]["comparison_struct"]["idx_best"] = sol.idx_best;
      json_log[fd]["run"][run][0]["comparison_struct"]["xs"] = xs;
      json_log[fd]["run"][run][0]["comparison_struct"]["mus"] = sol.mus;
      json_log[fd]["run"][run][0]["comparison_struct"]["sems"] = sol.sems;
      json_log[fd]["run"][run][0]["comparison_struct"]["ns"] = sol.ns;
      json_log[fd]["run"][run][0]["comparison_struct"]["pval"] = sol.pval;
      
      correct_inferences.push_back((sol.idx_best == correct_model ? 1.0 : 0.0));
      pvals.push_back(sol.pval);
      cout << "Model selected: " << sol.idx_best << " (Correct model: " << to_string(correct_model) << ")" << endl;
      cout << "p-val: " << sol.pval << endl;

    }
    catch(...) { // catch errors for logging/debugging and continue tests
      // std::exception_ptr p = std::current_exception();
      log.Log_Handler(string("Error in run ") + to_string(run) + string("\n") 
          // + string(p ? p.__cxa_exception_type()->name() : "null") + "\n" 
          + string("Check log.\n"));
      except_runs.push_back(run);
      // for (int w = 0; w < n_way; w++) {
      //   json_log[fd]["run"][run][w]["relative errors"][0] = nan("");
      // }
    }
  }

  double temp = 0;
  double num_reject = 0;
  for (int i = 0; i < correct_inferences.size(); i++) {
    temp += correct_inferences[i];
    if (pvals[i] < alpha && (correct_inferences[i] == 1.0)) { num_reject++; }
  }
  json_log[fd]["correct_model"] = correct_model;
  json_log[fd]["pvals"] = pvals;
  json_log[fd]["Proportion_correct"] = temp/((double)correct_inferences.size());
  json_log[fd]["power"] = num_reject/((double)pvals.size());
  cout << "Power for " << fd << " with mean difference: " << mean_diff << ": " << json_log[fd]["power"] << endl;

  if (except_runs.size() > 0) { 
    log.Log_Handler("Runs with exceptions occurred. See log.json for specific run numbers in item 'exception runs'.\n");
  }
  json_log[fd]["exception runs"] = except_runs;

  log.Log_Handler("\n");
  log.Flush_Handler();

  if (full_time_test && plotting) { plot_BO_sample_times(sample_times, test_func_str); }

  if (json_log[fd]["power"] < min_pass_prob) {
    fail += 1;
  }
  return fail;
}

#include <string>
#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"
#include "cnpy.h"
#include "sklearn_util.h"
#include "CNdlInterfaces.h"
#include <stdexcept>

#define DBG


int testGaussian(string type, string kernel, int verbosity);
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
    int verbose = 0;
    string kern;
    if (argc > 1) {
      command.setFlags(true);
      while (command.isFlags()) {
        if (command.isCurrentArg("-k", "--kernel")) {
          command.incrementArgument();
          kern = command.getCurrentArgument();
        }
        command.incrementArgument();
        if (command.isCurrentArg("-v", "--verbosity")) {
          command.incrementArgument();
          verbose = std::stoi(command.getCurrentArgument());
        }
        command.incrementArgument();
      }
      fail += testGaussian("ftc", kern, verbose);
    }
    else {
      fail += testGaussian("ftc", "poly", verbose);
      fail += testGaussian("ftc", "matern32", verbose);
      fail += testGaussian("ftc", "matern52", verbose);
      fail += testGaussian("ftc", "ratquad", verbose);
      fail += testGaussian("ftc", "rbf", verbose);
      //fail += testGaussian("dtc", "matern32", verbose);
      //fail += testGaussian("fitc", "matern32", verbose);
      //fail += testGaussian("pitc", "matern32", verbose);
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

int testGaussian(string type, string kernel, int verbosity)
{
  // TODO add testing with different optimization algorithms
  string fileName = "np_files" + ndlstrutil::dirSep() + "testSklearn_gpr_" + kernel + ".npz";  // + type + ".npz";
  int fail = 0;
  cnpy::npz_t npz_dict = cnpy::npz_load(fileName.c_str());
  double* temp = npz_dict["X"].data<double>();
  CMatrix X(temp, npz_dict["X"].shape[0], npz_dict["X"].shape[1]);

  // relative difference tolerance
  double tol = 1e-4;
  // absolute different tolerance
  double abs_tol = 1e-5;

  // CMatrix* test = readNpzFile(fileName, "X2");

  CMatrix y(npz_dict["y"].data<double>(), npz_dict["y"].shape[0], npz_dict["y"].shape[1]);
  int numActive = -1;  // no sparsity testing for now
  int approxType = 0;
  CMatrix scale = 1.0;
  CMatrix bias = 0.0;

  // TODO add another data set for testing

  CCmpndKern kern(X);
  CCmpndKern kern_ref(X);
  // parse .npy kernel parameters, obtain kernel structure
  getSklearnKernels(&kern, npz_dict, &X, true);
  getSklearnKernels(&kern_ref, npz_dict, &X, false);

  if (verbosity >= 0) {
    cout << "reference kern structure: " << endl << kern_ref.json_structure() << endl << endl;
  }

  // initialize model and reference model with parameters from sklearn
  CGaussianNoise noiseInit(&y);
  CGaussianNoise noiseInit_ref(&y);
  for(int i=0; i<noiseInit.getOutputDim(); i++)
  {
    noiseInit.setBiasVal(0.0, i);
    noiseInit_ref.setBiasVal(0.0, i);
  }
  noiseInit.setParam(0.0, noiseInit.getOutputDim());
  noiseInit_ref.setParam(0.0, noiseInit.getOutputDim());
  
  int iters = 100;
  bool outputScaleLearnt = false;
  string modelFileName = "testSklearn.model";

  kern.setBounds(kern_ref.getBounds());
  if (verbosity >= 0) {
    cout << "Kernel structure without reference params before optimization: " << endl;
    cout << kern.json_structure() << endl;
    cout << "Kernel state without reference params before optimization: " << endl;
    cout << kern.json_state() << endl << endl;
    for (int i = 0; i < kern.getNumParams(); i++) {
      cout << i << " " << kern.getParamName(i) << endl;
    }
  }
  CGp model(&kern, &noiseInit, &X, approxType, numActive, verbosity);
  CGp model_ref(&kern_ref, &noiseInit_ref, &X, approxType, numActive, verbosity);

  model.setBetaVal(1);
  model_ref.setBetaVal(1);
  model.setScale(scale);
  model_ref.setScale(scale);
  model.setBias(bias);
  model_ref.setBias(bias);
  temp = npz_dict["obsNoiseVar"].data<double>();
  model.setObsNoiseVar(*temp);
  model_ref.setObsNoiseVar(*temp);
  model.updateM();
  model_ref.updateM();

  model.setVerbosity(verbosity);
  int default_optimiser = CGp::LBFGS_B;
  model.setDefaultOptimiser(default_optimiser);  // options: LBFGS_B, BFGS, SCG, CG, GD
  // (roughly) match sklearn tolerances for L-BFGS-B.
  model.setObjectiveTol(1e-5);
  model.setParamTol(2.22e-9);
  model.setOutputScaleLearnt(outputScaleLearnt);
  model.optimise(iters);

  // string comment = "";
  // writeGpToFile(model, modelFileName, comment);
  
  // // Initialise a model from gpInfo.
  // CGp model(&X, &y, &kern, &noiseInit, fileName.c_str(), "gpInfoInit", 0);
  // if(model.equals(modelInit))
  // { 
  //   cout << "GP initial model passed." << endl;
  // }
  // else 
  // {
  //   cout << "FAILURE: GP." << endl;
  //   cout << "Matlab loaded model " << endl;
  //   model.display(cout);
  //   cout << "C++ Initialised Model " << endl;
  //   modelInit.display(cout);
  //   fail++;
  // }
  // */

  cout << endl << "Test of GPc with kernel: " << kernel << endl << endl;

  // Compare optimized C++ parameters with MATLAB provided parameters.
  CMatrix params_ref(1, model_ref.getOptNumParams());
  // fileParams.readMatlabFile(fileName.c_str(), "params");
  CMatrix params(1, model.getOptNumParams());

  double diff;
  double abs_diff;

  // tests passed:
  // most direct comparisons passed in kernels and model training.
  // close to passing for most other comparisons
  // qualitative behavior matching in all cases but for polynomial kernel with current data set
  // MATLAB tests by authors presumably passed?

  // OUTSTANDING ISSUES
  // rationalquadratic in particular may be running into numerical stability issues with alpha parameter, 
  //    giving qualitatively similar behavior
  // RBF giving completely different gradient values for random initial parameters (optimized gradients are roughly matching)
  // poly kernel failing to converge, returning non-PSD errors (sklearn had no issues)
  // poly kernel also returning values off by ~16000 in kernel test

  // everything else is matching or roughly matching even if a comparison is failing my particular choices for tolerances

  // TODO
  // check Nia tests
  // identical starting values
  //    comparison of initial gradient values gives a partial test of this
  // using different optimization algorithms currently.
  //    try using different optimization algorithms in GCp, see variability of results
  //        roughly similar results qualitatively, same tests passing and failing, similar values. 
  //        alpha values in ratquad different, further suggests numerical instability of that parameter for this data set
  // clean up test code, put in place options to test with sklearn or MATLAB, submit pull request, put on github

  // DONE (roughly) set optimizers to have identical tolerances, optimizers are using incompatible convergence criteria
  // DONE use identical noise model variances
  // DONE compare inference values for fixed hyperparameter choices
  //    inferences closer (10x closer wrt max absolute difference) than with GCp-fit parameters as expected 
  //    but still not within floating point error. 
  //    Absolute differences larger than absolute differences in kernel comparisons (which never had relative comparisons)

  // try different data sets, this one might be too simple to see significant differences between kernels, which 
  // appear to be giving almost indistinguishable answers (e.g. rat-quad vs. rbf)

  // play around with different kernels, hyperparameters, datasets in general, just try setting the hyperparameters directly

  // compare optimized parameters and reference parameters
  model.getOptParams(params);
  model_ref.getOptParams(params_ref);
  diff = params.maxRelDiff(params_ref);
  if(diff < tol)
  {
    cout << "PASSED: Parameter match passed. " << endl;
    cout << "Max relative difference: " << diff << endl << endl;
  }
  else
  {
    cout << "FAILURE: GP parameter match." << endl;
    cout << "Maximum relative difference: " << diff << endl;
    cout << "Parameters (reference parameters followed by C++ optimized parameters):" << endl;
    for (int i = 0; i < params_ref.getCols(); i++) {
      cout << model.pkern->getParamName(i) << ": " << params_ref.getVal(0, i) << ",   " << params.getVal(0, i) << endl;
    }
    cout << endl;
    fail++;
  }
  
  // Compare GCp gradients (from reference model to have same parameters) with reference implementation gradients.
  CMatrix grads_ref(npz_dict["gradTheta"].data<double>(), 1, model_ref.getOptNumParams());
  // grads_ref.readMatlabFile(fileName.c_str(), "grads");
  CMatrix grads(1, model_ref.getOptNumParams());
  double L = model_ref.logLikelihoodGradient(grads);

  // cout << "Kernel with reference params: " << endl;
  // cout << model_ref.pkern->json_structure() << endl;
  // cout << model_ref.pkern->json_state() << endl;

  // cout << grads << endl;
  // cout << grads_ref << endl;

  // use absolute difference for optimized gradients which are approaching zero
  // and may be expected to have larger absolute differences
  diff = grads.maxRelDiff(grads_ref);
  abs_diff = grads.maxAbsDiff(grads_ref);
  if(diff < abs_tol)
  {
    cout << "PASSED: GP gradient match passed with reference parameters." << endl;
    cout << "Max relative difference: " << diff << endl;
    cout << "Max absolute difference: " << abs_diff << endl << endl;
  }
  else {
    cout << "FAILURE: GP optimized gradient match with reference parameters." << endl;
    cout << "Reference gradient:" << endl << grads_ref;
    cout << "C++ Gradient:" << endl << grads;
    cout << "Max relative difference: " << diff << endl;
    cout << "Max absolute difference: " << abs_diff << endl << endl;
    fail++;
  }

  CMatrix K(X.getRows(), X.getRows());
  model_ref.pkern->compute(K, X);

  CMatrix K_ref(npz_dict["K"].data<double>(), npz_dict["K"].shape[0], npz_dict["K"].shape[1]);

  diff = K.maxRelDiff(K_ref);
  abs_diff = K.maxAbsDiff(K_ref);
  if(diff < tol) {
    cout << "PASSED: " << model_ref.pkern->getName() << " inference kernel covar compute matches with reference parameters" << endl;
    cout << "Matches within " << diff  << " max relative difference." << endl << endl;
  }
  else { 
    cout << "FAILURE: " << model_ref.pkern->getName() << " full inference kernel compute with reference parameters." << endl;
    cout << "Maximum relative difference: " << diff << endl;    
    cout << "Maximum absolute difference: " << abs_diff << endl << endl;
    fail++;
  }

  // CMatrix K(X.getRows(), X.getRows());
  // model_ref.pkern->compute(K, X);

  // model_ref.display(cout);
  // cout << "CGp K updated: " << model_ref.isKupToDate() << endl;

  CMatrix K_train(model_ref.K);

  CMatrix K_train_ref(npz_dict["K_train"].data<double>(), npz_dict["K_train"].shape[0], npz_dict["K_train"].shape[1]);

  diff = K_train.maxAbsDiff(K_train_ref);
  if(diff < tol) {
    cout << "PASSED: " << model_ref.pkern->getName() << " train covar compute matches within " << diff  << " max absolute difference." << endl << endl;
  }
  else {
    cout << "FAILURE: " << model_ref.pkern->getName() << " full train kernel compute." << endl;
    cout << "Maximum absolute difference: " << diff << endl << endl;
    fail++;
  }

  // Compare C++ log likelihood (with reference implementation parameters) with reference implementation log likelihood.
  CMatrix logL_ref(npz_dict["logL"].data<double>(), 1, 1);
  CMatrix logL(1, 1);
  logL.setVal(model_ref.logLikelihood(), 0, 0);
  diff = logL.maxRelDiff(logL_ref);
  if(diff < tol)
  {
    cout << "PASSED: GP Log Likelihood match with reference parameters." << endl;
    cout << "Max relative difference: " << diff << endl << endl;
  }
  else
  {
    cout << "FAILURE: GP Log Likelihood match with reference parameters." << endl;
    cout << "Reference Log Likelihood: " << logL_ref;
    cout << "C++ Log Likelihood: " << logL;
    cout << "Max relative difference: " << diff << endl << endl;
    fail++;
  }

  // Compare optimized C++ log likelihood with reference implementation log likelihood.
  // somewhat unnecessary
  logL.setVal(model.logLikelihood(), 0, 0);
  diff = logL.maxRelDiff(logL_ref);
  if(diff < tol)
  {
    cout << "PASSED: GP Log Likelihood match." << endl;
    cout << "Max relative difference: " << diff << endl << endl;
  }
  else
  {
    cout << "FAILURE: GP Log Likelihood match." << endl;
    cout << "Reference Log Likelihood: " << logL_ref;
    cout << "C++ Log Likelihood: " << logL;
    cout << "Max relative difference: " << diff << endl << endl;
    fail++;
  }

  // compare GCp inferences (GCp-fit parameters) with reference implementation inferences
  CMatrix X_pred(npz_dict["X_pred"].data<double>(), npz_dict["X_pred"].shape[0], npz_dict["X_pred"].shape[1]);
  CMatrix y_pred_ref(npz_dict["y_pred"].data<double>(), npz_dict["y_pred"].shape[0], 1);
  CMatrix std_pred_ref(npz_dict["std_pred"].data<double>(), npz_dict["std_pred"].shape[0], 1);
  CMatrix y_pred(X_pred.getRows(), 1);
  CMatrix std_pred(X_pred.getRows(), 1);
  model.out(y_pred, std_pred, X_pred);
  diff = y_pred.maxRelDiff(y_pred_ref);
  abs_diff = y_pred.maxAbsDiff(y_pred_ref);
  if(diff < tol) { cout << "PASSED: GP y-prediction match. " << endl; }
  else 
  {
    cout << "FAILURE: GP y prediction match." << endl;
    fail++;
  }
  cout << "Max absolute difference: " << abs_diff << endl;
  cout << "Max relative difference: " << diff << endl << endl;

  diff = std_pred.maxRelDiff(std_pred_ref);
  if(diff < tol) { cout << "PASSED: GP standard deviation match." << endl; }
  else 
  {
    cout << "FAILURE: GP standard deviation match." << endl;
    fail++;
  }
  cout << "Max relative difference: " << diff << endl << endl;
  
  // compare GCp inferences (using reference parameters) with reference implementation inferences
  model_ref.out(y_pred, std_pred, X_pred);
  diff = y_pred.maxRelDiff(y_pred_ref);
  abs_diff = y_pred.maxAbsDiff(y_pred_ref);
  if(diff < tol)
  { cout << "PASSED: GP y-prediction (using reference parameters) match. " << endl; }
  else 
  {
    cout << "FAILURE: GP y-prediction (using reference parameters) match." << endl;
    fail++;
  }
  cout << "Max absolute difference: " << abs_diff << endl;
  cout << "Max relative difference: " << diff << endl << endl;

  diff = std_pred.maxRelDiff(std_pred_ref);
  if(diff < tol) { cout << "PASSED: GP standard deviation (using reference parameters) match. " << endl; }
  else 
  {
    cout << "FAILURE: GP standard deviation (using reference parameters) match." << endl;
    fail++;
  }
  cout << "Max relative difference: " << diff << endl << endl;

  // Compare GCp gradients with random parameters with reference implementation gradients.
  CMatrix grads_rand_ref(npz_dict["gradTheta_rand"].data<double>(), 1, model.getOptNumParams());
  CMatrix params_rand(npz_dict["theta_rand"].data<double>(), 1, model.getOptNumParams());
  model_ref.setOptParams(params_rand);
  model_ref.logLikelihoodGradient(grads);
  diff = grads.maxRelDiff(grads_rand_ref);
  if(diff < tol)
  {
    cout << "PASSED: GP random parameter gradient match." << endl;
    cout << "Max relative difference: "  << diff << endl << endl;
  }
  else {
    cout << "FAILURE: GP random parameter gradient match." << endl;
    cout << "Reference gradient:" << endl << grads_rand_ref;
    cout << "C++ Gradient:" << endl << grads;
    cout << "Max relative difference: " << diff << endl << endl;
    fail++;
  }

  // attempt at comparing covariance gradients between sklearn and CGp
  // sklearn computes kernel gradients as grad of covar wrt log-hps
  //   CGp computes covGrad as grad of log likelihood wrt covariance 
  //     kernel covariance is grad of LL wrt hps, can compute with all-ones gradCov (grads of transformed params),
  //     compare with sum of sklearn kernel grads  // CMatrix gKX(temp, npz_dict["gXX"].shape[0], npz_dict["gKX"].shape[1]);
  // model_ref.covGrad;

  // CMatrix K_ref(npz["K"].data<double>(), npz["K"].shape[0], npz["K"].shape[1]);

  // diff = K.maxAbsDiff(K_ref);
  // if(diff < tol)
  //   cout << model_ref.pkern->getName() << " full compute matches within " << diff  << " max absolute difference." << endl;
  // else
  // { 
  //   cout << "FAILURE: " << model_ref.pkern->getName() << " full compute." << endl;
  //   cout << "Maximum absolute difference: " << diff << endl;    
  //   fail++;
  // }

  // // Read and write tests
  // // Matlab read/write
  // // model.writeMatlabFile("crap.mat", "writtenModel");
  // // modelInit.readMatlabFile("crap.mat", "writtenModel");
  // if(model.equals(modelInit))
  //   cout << "MATLAB written " << model.getName() << " matches read in model. Read and write to MATLAB passes." << endl;
  // else
  // {
  //   cout << "FAILURE: MATLAB read in " << model.getName() << " does not match written out model." << endl;
  //   fail++;
  // }

  // // Write to stream.
  // // model.toFile("crap_gp");
  // // modelInit.fromFile("crap_gp");
  // if(model.equals(modelInit))
  // {
  //   cout << "Text written " << model.getName() << " matches read in model. Read and write to stream passes." << endl;
  // }
  // else
  // {
  //   cout << "FAILURE: Stream read in " << model.getName() << " does not match written model." << endl;
  //   cout << "Matlab loaded model " << endl;
  //   model.display(cout);
  //   cout << "Text Read in Model " << endl;
  //   modelInit.display(cout);
  //   fail++;
  // }
  
  // //  model.checkGradients();

  return fail;
}


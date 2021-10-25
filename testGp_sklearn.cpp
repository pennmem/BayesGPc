#include <string>
#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"
#include "cnpy.h"
#include "sklearn_util.h"
#include "CNdlInterfaces.h"
#include <stdexcept>


int testGaussian(string type, string kernel);
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
    if (argc > 1) {
      command.setFlags(true);
      while (command.isFlags()) {
        string arg = command.getCurrentArgument();
        if (command.isCurrentArg("-k", "--kernel")) {
          command.incrementArgument();
          arg = command.getCurrentArgument();
          fail += testGaussian("ftc", arg);
        }
        command.incrementArgument();
      }
    }
    else {
      // fail += testGaussian("ftc", "matern32");
      // fail += testGaussian("ftc", "matern52");
      // fail += testGaussian("ftc", "poly");
      fail += testGaussian("ftc", "ratquad");
      // fail += testGaussian("ftc", "rbf");
      //fail += testGaussian("dtc");
      //fail += testGaussian("fitc");
      //fail += testGaussian("pitc");
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

int testGaussian(string type, string kernel)
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

  // for (int i=0; i < 5; i++){
  //   cout << temp[i] << " " << X.getVal(i) << endl;
  // }

  CMatrix y(npz_dict["y"].data<double>(), npz_dict["y"].shape[0], npz_dict["y"].shape[1]);
  // y.readMatlabFile(fileName.c_str(), "y");
  // CMatrix actSetMat;
  // actSetMat.readMatlabFile(fileName.c_str(), "numActive");
  // int numActive = (int)actSetMat.getVal(0);
  int numActive = -1;  // no sparsity testing for now
  // CMatrix apprxTypeMat;
  // apprxTypeMat.readMatlabFile(fileName.c_str(), "approxInt");
  // int approxType = (int)apprxTypeMat.getVal(0); 
  int approxType = 0;
  CMatrix scale = 1.0;
  // scale.readMatlabFile(fileName.c_str(), "scale");
  CMatrix bias = 0.0;
  // bias.readMatlabFile(fileName.c_str(), "bias");

  // TODO what is GaussianNoise?
  // TODO how are gradients computed? following the references back to getGradParam for most of the kernels 
  //          leads to dead ends where most of those functions aren't implemented, including for Mater32,
  //          which is clearly working, so just follow back the execution with break points to see where the
  //          where the gradients are actually being computed
  // TODO what is learning the output scale?
  // TODO add another data set for testing
  // TODO add hyperparameter gradient testing to GPc testing (not sure if covariance gradient can be easily pulled out
  //      from sklearn models, not really necessary if gradients match otherwise)
  // TODO finish GPc testing
  // TODO switch kernel tests to relative absolute error? would still fail for small values with imprecision...

  // TODO compare GPR inferences, stds, marginal log-likelihoods, gradients for specified input parameters
  // TODO find unit test library for C++

  CCmpndKern kern(X);
  CCmpndKern kern_ref(X);
  // parse .npy kernel parameters, obtain kernel structure
  bool structureOnly = false;
  bool arbitraryChange = true;
  getSklearnKernels(&kern, npz_dict, &X, true);
  getSklearnKernels(&kern_ref, npz_dict, &X, false);

  // initialize model and reference model with parameters from sklearn
  CGaussianNoise noiseInit(&y);
  CGaussianNoise noiseInit_ref(&y);
  for(int i=0; i<noiseInit.getOutputDim(); i++)
  {
    noiseInit.setBiasVal(0.0, i);
    noiseInit_ref.setBiasVal(0.0, i);
  }
  noiseInit.setParam(1e-10, noiseInit.getOutputDim());
  noiseInit_ref.setParam(1e-10, noiseInit.getOutputDim());
  
  int iters = 100;
  bool outputScaleLearnt = false;
  string modelFileName = "testSklearn.model";

  CGp model(&kern, &noiseInit, &X, approxType, numActive, 3);
  CGp model_ref(&kern_ref, &noiseInit_ref, &X, approxType, numActive, 3);

  model.setBetaVal(1);
  model_ref.setBetaVal(1);
  model.setScale(scale);
  model_ref.setScale(scale);
  model.setBias(bias);
  model_ref.setBias(bias);
  model.updateM();
  model_ref.updateM();

  model.setVerbosity(2);
  model.setDefaultOptimiser(CGp::BFGS);  //options: SCG, CG, GD
  // (roughly) match sklearn tolerances for L-BFGS-B.
  model.setObjectiveTol(1e-5);
  model.setParamTol(2.22e-9);
  model.setOutputScaleLearnt(outputScaleLearnt);
  model.optimise(iters);

  string comment = "";
  writeGpToFile(model, modelFileName, comment);
  
  // /* loading in model from MATLAB
  // CCmpndKern kern(X);
  // kern.readMatlabFile(fileName.c_str(), "kernInit");
  // //CGaussianNoise noise(y);
  // //noise.readMatlabFile("testGp.mat", "noiseInit");

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

  // everything else is matching or roughyl matching even if a comparison is failing my particular choices for tolerances

  // TODO
  // check Nia tests
  // add relative error comparisons to kernel tests
  // identical starting values
  //    comparison of initial gradient values gives a partial test of this
  // using different optimization algorithms currently.
  //    GCp is using LBFGS (unconstrained optimization) vs. sklearn with L-BFGS-B (box constrained, different algo, 
  //        more recent implementation out of same lab at Northwestern)
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
  // write down the algorithm for tuning GPR hyperparameters

  model.getOptParams(params);
  model_ref.getOptParams(params_ref);
  double diff = params.maxRelDiff(params_ref);
  if(diff < tol)
  {
    cout << "Parameter match passed. " << endl;
    cout << "Max relative difference: " << diff << endl << endl;
  }
  else
  {
    cout << "FAILURE: GP parameter match." << endl;
    cout << "Reference implementation loaded params: " << endl << params_ref;
    cout << "C++ optimised params: " << endl << params;
    cout << "Maximum relative difference: " << params.maxRelDiff(params_ref) << endl << endl;
    fail++;
  }
  
  // Compare GCp gradients (from reference model to have same parameters) with reference implementation gradients.
  CMatrix grads_ref(npz_dict["gradTheta"].data<double>(), 1, model_ref.getOptNumParams());
  // grads_ref.readMatlabFile(fileName.c_str(), "grads");
  CMatrix grads(1, model_ref.getOptNumParams());
  model_ref.logLikelihoodGradient(grads);

  cout << grads;
  cout << grads_ref << endl;

  // use absolute difference for optimized gradients which are approaching zero
  // and may be expected to have larger absolute differences
  diff = grads.maxRelDiff(grads_ref);
  if(diff < abs_tol)
  {
    cout << "GP optimized gradient match passed." << endl;
    cout << "Max absolute difference: " << diff << endl;
    cout << "Max relative difference: " << grads.maxRelDiff(grads_ref) << endl << endl;
  }
  else {
    cout << "FAILURE: GP optimized gradient match." << endl;
    cout << "Reference gradient:" << endl << grads_ref;
    cout << "C++ Gradient:" << endl << grads;
    cout << "Max relative difference: " << grads.maxRelDiff(grads_ref) << endl;
    cout << "Max absolute difference: " << diff << endl << endl;
    fail++;
  }

  // Compare C++ log likelihood with reference implementation log likelihood.
  CMatrix logL_ref(npz_dict["logL"].data<double>(), 1, 1);
  // logL_ref.readMatlabFile(fileName.c_str(), "ll");
  CMatrix logL(1, 1);
  logL.setVal(model.logLikelihood(), 0, 0);
  diff = logL.maxRelDiff(logL_ref);
  if(diff < tol)
  {
    cout << "GP Log Likelihood match passed. " << endl;
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
  double abs_diff = y_pred.maxAbsDiff(y_pred_ref);
  if(diff < tol)
  {
    cout << "GP y-prediction match passed. " << endl;
  }
  else 
  {
    cout << "FAILURE: GP y prediction match." << endl;
    fail++;
  }
  cout << "Max absolute difference: " << abs_diff << endl;
  cout << "Max relative difference: " << diff << endl << endl;

  diff = std_pred.maxRelDiff(std_pred_ref);
  if(diff < tol)
  {
    cout << "GP standard deviation match passed. " << endl;
  }
  else 
  {
    cout << "FAILURE: GP standard deviation match." << endl;
    fail++;
  }
  cout << "Max relative difference: " << diff << endl << endl;
  
  // compare GCp inferences (sklearn-fit parameters) with reference implementation inferences
  model.setOptParams(params_ref);
  model.out(y_pred, std_pred, X_pred);

  diff = y_pred.maxRelDiff(y_pred_ref);
  abs_diff = y_pred.maxAbsDiff(y_pred_ref);
  if(diff < tol)
  {
    cout << "GP y-prediction (using reference parameters) match passed. " << endl;
  }
  else 
  {
    cout << "FAILURE: GP y-prediction (using reference parameters) match." << endl;
    fail++;
  }
  cout << "Max absolute difference: " << abs_diff << endl;
  cout << "Max relative difference: " << diff << endl << endl;

  diff = std_pred.maxRelDiff(std_pred_ref);
  if(diff < tol)
  {
    cout << "GP standard deviation (using reference parameters) match passed. " << endl;
  }
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
    cout << "GP random parameter gradient match passed." << endl;
    cout << "Max relative difference: "  << diff << endl << endl;
  }
  else {
    cout << "FAILURE: GP random parameter gradient match." << endl;
    cout << "Reference gradient:" << endl << grads_ref;
    cout << "C++ Gradient:" << endl << grads;
    cout << "Max relative difference: " << diff << endl << endl;
    fail++;
  }

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



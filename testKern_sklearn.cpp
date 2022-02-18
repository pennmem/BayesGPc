#include "CKern.h"
#include "cnpy.h"
#include "sklearn_util.h"

using namespace std;
using namespace nlohmann;

int testType(const string kernelType);
int testKern(CKern* kern, const string fileName);
int testKernNaming();

int main()
{
  int fail=0;
  try
  {
    // fail += testType("white");
    // // fail += testType("bias");

    // fail += testType("matern32");
    // fail += testType("matern52");

    // fail += testType("rbf");
    // fail += testType("ratquad");
    // fail += testType("poly");  // TODO not currently matching sklearn

    // // fail += testType("lin");

    // // fail += testType("mlp");
    // //fail += testType("gibbs");

    //fail += testType("rbfperiodic");
    //fail += testType("gibbsperiodic");
 
    // fail += testType("rbfard");
    // fail += testType("ratquadard");
    // fail += testType("matern32ard");
    // fail += testType("matern52ard");

    // fail += testType("linard");
    // fail += testType("polyard");
    // fail += testType("mlpard");
    // fail += testType("gibbsard");
   
    // fail += testType("cmpnd");
    // // fail += testType("tensor");
    fail += testKernNaming();
    cout << "Number of failures: " << fail << "." << endl;
  }
  catch(ndlexceptions::FileFormatError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileReadError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileWriteError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::FileError err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(ndlexceptions::Error err)
  {
    cerr << err.getMessage();
    exit(1);
  }
  catch(std::bad_alloc err)
  {
    cerr << "Out of memory.";
    exit(1);
  }
  catch(std::exception err)
  {
    cerr << "Unhandled exception.";
    exit(1);
  }
}

int testType(const string kernelType)
{
  string fileName = "np_files" + ndlstrutil::dirSep() + "testSklearn_kernel_" + kernelType + ".npz";
  cnpy::npz_t npz = cnpy::npz_load(fileName.c_str());

  double* temp = npz["X"].data<double>();
  CMatrix X(temp, npz["X"].shape[0], npz["X"].shape[1]);
  // X.readMatlabFile(fileName, "X");

  CKern* kern;
  bool structureOnly = false;
  bool arbitraryChange = true;
  string param_key = "k1";
  string kernel_key;
  CKern* kern2;
 
  if(kernelType=="matern32")
  {
    kernel_key = "Matern";
  }
  if(kernelType=="white")
  {
    kernel_key = "WhiteKernel";
  }
  // else if(kernelType=="bias")
  // {
  //   kern = new CBiasKern(X);
  //   kern2 = new CBiasKern(X);
  // }
  else if(kernelType=="rbf")
  {
    kernel_key = "RBF";
  }
  else if(kernelType=="ratquad")
  {
    kernel_key = "RationalQuadratic";
  }
  else if(kernelType=="matern32")
  {
    kernel_key = "Matern";
  }
  else if(kernelType=="matern52")
  {
    kernel_key = "Matern";
  }
  // else if(kernelType=="lin")
  // {
  //   kern = new CLinKern(X);
  //   kern2 = new CLinKern(X);
  // }
  else if(kernelType=="poly")
  {
    kernel_key = "DotProduct";
  }
  // else if(kernelType=="mlp")
  // {
  //   kern = new CMlpKern(X);
  //   kern2 = new CMlpKern(X);
  // }
  // // else if(kernelType=="gibbs")
  // // { 
  // //   kern = new CGibbsKern(X);
  // //   kern2 = new CGibbsKern(X);
  // // }
  // // else if(kernelType=="rbfperiodic")
  // // { 
  // //   kern = new CRbfPeriodicKern(X);
  // //   kern2 = new CRbfPeriodicKern(X);
  // // }
  // // else if(kernelType=="gibbsperiodic")
  // // { 
  // //   kern = new CGibbsPeriodicKern(X);
  // //   kern2 = new CGibbsPeriodicKern(X);
  // // }
  // else if(kernelType=="rbfard")
  // {
  //   kern = new CRbfardKern(X);
  //   kern2 = new CRbfardKern(X);
  // }
  // // else if(kernelType=="ratquadard")
  // // { 
  // //   kern = new CRatQuadardKern(X);
  // //   kern2 = new CRatQuadardKern(X);
  // // }
  // // else if(kernelType=="matern32ard")
  // // { 
  // //   kern = new CMatern32ardKern(X);
  // //   kern2 = new CMatern32ardKern(X);
  // // }
  // // else if(kernelType=="matern52ard")
  // // { 
  // //   kern = new CMatern52ardKern(X);
  // //   kern2 = new CMatern52ardKern(X);
  // // }
  // else if(kernelType=="linard")
  // {
  //   kern = new CLinardKern(X);
  //   kern2 = new CLinardKern(X);
  // }
  // else if(kernelType=="polyard")
  // {
  //   kern = new CPolyardKern(X);
  //   kern2 = new CPolyardKern(X);
  // }
  // else if(kernelType=="mlpard")
  // {
  //   kern = new CMlpardKern(X);
  //   kern2 = new CMlpardKern(X);
  // }
  // // else if(kernelType=="gibbsard")
  // // { 
  // //   kern = new CGibbsardKern(X);
  // //   kern2 = new CGibbsardKern(X);
  // // }

  // else if(kernelType=="cmpnd")
  // {
  //   kern = new CCmpndKern(X);
    
  //   // kern->addKern(new CRatQuadKern(X));
  //   kern->addKern(new CRbfKern(X));
  //   // kern->addKern(new CLinKern(X));
  //   // kern->addKern(new CPolyKern(X));
  //   // kern->addKern(new CBiasKern(X));
  //   kern->addKern(new CWhiteKern(X));

  //   // not implemented in sklearn currently
  //   // kern->addKern(new CRbfardKern(X));
  //   // kern->addKern(new CMlpardKern(X));
  //   // kern->addKern(new CLinardKern(X));
  //   // kern->addKern(new CPolyardKern(X));
  //   // kern->addKern(new CMlpKern(X));
    
  //   kern2 = new CCmpndKern(X);
  // }
  // else if(kernelType=="tensor")
  // {
  //   kern = new CTensorKern(X);
  //   kern->addKern(new CRbfKern(X));
  //   kern->addKern(new CLinKern(X));
  //   kern->addKern(new CPolyKern(X));
  //   kern2 = new CTensorKern(X);
  // }
  else
  {
    throw ndlexceptions::Error("Unrecognised kernel type requested.");
  }
  kern = getSklearnKernel(X.getCols(), npz, kernel_key, param_key, false);
  // kern2 not used currently
  // kern2 = getSklearnKernel(X.getCols(), npz, kernel_key, param_key, false);

  int fail = testKern(kern, fileName);
  delete kern;
  return fail;
}

int testKern(CKern* kern, const string fileName)
{
  int fail = 0;
  cnpy::npz_t npz = cnpy::npz_load(fileName.c_str());

  double tol = 1e-6;
  cout << "Test absolute max difference tolerance: " << tol << endl;

  // double* temp = npz["params"].data<double>();
  // CMatrix params(temp, npz["params"].shape[0], npz["params"].shape[1]);

  // temp = npz["X"].data<double>();
  CMatrix X(npz["X"].data<double>(), npz["X"].shape[0], npz["X"].shape[1]);
  CMatrix X2(npz["X2"].data<double>(), npz["X2"].shape[0], npz["X2"].shape[1]);
  // kern->setTransParams(params);
  // test GCp model and model loaded directly from reference implementation match
  // kern2->readMatlabFile(fileName, "kern2");
  // if(kern2->equals(*kern))
  //   cout << kern->getName() << " Initial Kernel matches." << endl;
  // else
  // {
  //   cout << "FAILURE: " << kern->getName() << " Initial Kernel." << endl;
  //   cout << "Matlab kernel" << endl << *kern2 << endl << "C++ Kernel " << endl << *kern << endl;
  //   fail++;
  // }
  // test GCp and reference kernel matrix computations match
  CMatrix K(X.getRows(), X.getRows());
  kern->compute(K, X);

  CMatrix K_ref(npz["K"].data<double>(), npz["K"].shape[0], npz["K"].shape[1]);
  // if(K.equals(K_ref))
  // cout << "K" << endl << K;
  // cout << "K_ref" << endl << K_ref;

  double diff;
  diff = K.maxAbsDiff(K_ref);
  if(diff < tol)
    cout << kern->getName() << " full compute matches within " << diff  << " max absolute difference." << endl;
  else
  { 
    cout << "FAILURE: " << kern->getName() << " full compute." << endl;
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }

  // test cross kernels
  CMatrix K1_2(X.getRows(), X2.getRows());
  kern->compute(K1_2, X, X2);
  CMatrix K1_2_ref(npz["K1_2"].data<double>(), npz["K1_2"].shape[0], npz["K1_2"].shape[1]);
  diff=K1_2.maxAbsDiff(K1_2_ref);
  if(diff < tol)
    cout << kern->getName() << " double compute matches within " << diff  << " max absolute difference." << endl;
  else
  { 
    cout << "FAILURE: " << kern->getName() << " double compute." << endl;
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }
  // test diagonal kernel computations match
  CMatrix K_diag(X.getRows(), 1);
  kern->diagCompute(K_diag, X);
  CMatrix K_diag_ref(npz["K_diag"].data<double>(), npz["K_diag"].shape[0], npz["K_diag"].shape[1]);
  diff=K_diag.maxAbsDiff(K_diag_ref);
  if(diff < tol)
    cout << kern->getName() << " diag compute matches within " << diff  << " max absolute difference." << endl;
  else
  {
    cout << "FAILURE: " << kern->getName() << " diag compute." << endl;
    cout << "Maximum absolute difference: " << diff << endl;    
    fail++;
  }

  // TODO would like to compare covariance gradients, 
  // however GCp doesn't seem to have a simple algorithm for computing d-kernel_params/dK
  // following code gives partial attempt at doing this, gradients needed as mentioned aren't currently computed
  // gradients can be tested with full GPR model, which is done in testGp_sklearn.cpp

  // CMatrix covGrad(npz["gKX"].data<double>(), npz["gKX"].shape[0], npz["gKX"].shape[1]);
  // // covGrad.readMatlabFile(fileName, "covGrad");
  // covGrad.setSymmetric(true);
  // CMatrix g1(1, kern->getNumParams());
  // kern->getGradTransParams(g1, X, covGrad);
  // CMatrix g2;
  // CMatrix g2(npz["gKX"].data<double>(), npz["gKX"].shape[0], npz["gKX"].shape[1]);
  // // g2.readMatlabFile(fileName, "g2");
  // if(g1.equals(g2))
  //   cout << kern->getName() << " parameter gradient matches." << endl;
  // else
  // {
  //   cout << "FAILURE: " << kern->getName() << " parameter gradient." << endl;
  //   double diff=g1.maxAbsDiff(g2);
  //   cout << "Matlab gradient: " << endl;
  //   cout << g2 << endl;
  //   cout << "C++ gradient: " << endl;
  //   cout << g1 << endl;
  //   cout << "Maximum absolute difference: " << diff << endl;    
  //   fail++;
  // }

  // // test cross kernel gradients match
  // CMatrix covGrad2;
  // covGrad2.readMatlabFile(fileName, "covGrad2");
  // covGrad.setSymmetric(true);
  // CMatrix g3(1, kern->getNumParams());
  // kern->getGradTransParams(g3, X, X2, covGrad2);
  // CMatrix g4;
  // g4.readMatlabFile(fileName, "g4");
  // if(g3.equals(g4))
  //   cout << kern->getName() << " parameter X, X2 gradient matches." << endl;
  // else
  // {
  //   cout << "FAILURE: " << kern->getName() << " parameter X, X2 gradient." << endl;
  //   double diff=g3.maxAbsDiff(g4);
  //   cout << "Matlab gradient: " << endl;
  //   cout << g4 << endl;
  //   cout << "C++ gradient: " << endl;
  //   cout << g3 << endl;
  //   cout << "Maximum absolute difference: " << diff << endl;    
  //   fail++;
  // }
  
  // /* Gradient with respect to X */
  // vector<CMatrix*> G2;
  // MATFile* matFile=matOpen(fileName.c_str(), "r");
  // if(matFile==NULL)
  //   throw ndlexceptions::FileReadError(fileName);
  // mxArray* matlabArray = matGetVariable(matFile, "G2");
  // if(matlabArray==NULL)
  //   throw ndlexceptions::MatlabInterfaceReadError("G2");
  // if(!mxIsCell(matlabArray))
  //   throw ndlexceptions::MatlabInterfaceReadError("G2");
  // int elems = mxGetNumberOfElements(matlabArray);
  // for(int i=0; i<elems; i++)
  // {
  //   mxArray* matlabMatrix = mxGetCell(matlabArray, i);
  //   G2.push_back(new CMatrix(1, 1));
  //   G2[i]->fromMxArray(matlabMatrix);
  // }
  // vector<CMatrix*> gX;
  // for(int i=0; i<X.getRows(); i++)
  //   gX.push_back(new CMatrix(X2.getRows(), X2.getCols()));
  // kern->getGradX(gX, X, X2);
  // double maxDiff=0.0;
  // for(int i=0; i<gX.size(); i++)
  // {
  //   double diff=gX[i]->maxAbsDiff(*G2[i]);
  //   if(diff>maxDiff)
  //     maxDiff=diff;
  // }
  // if(maxDiff<ndlutil::MATCHTOL)
  //   cout << kern->getName() << " X gradient matches." << endl;
  // else
  // {
  //   cout << "FAILURE: " << kern->getName() << " gradient with respect to X does not match." << endl;
  //   cout << "Maximum absolute difference: " << maxDiff << endl;
  //   fail++;
  // }
  
  // /* Gradient with respect to diagonal X */
  // CMatrix GD2(X.getRows(), X.getCols());
  // GD2.readMatlabFile(fileName, "GD2");
  // CMatrix gDX(X.getRows(), X.getCols());
  // kern->getDiagGradX(gDX, X);
  // if(GD2.equals(gDX))
  //   cout << kern->getName() << " diagonal X gradient matches." << endl;
  // else
  // {
  //   cout << "FAILURE: " << kern->getName() << " gradient of diagonal with respect to X does not match." << endl;
  //   cout << "Maximum absolute difference: " << GD2.maxAbsDiff(gDX) << endl;
  //   fail++;
  // }
  
  
  
  // // Read and write tests
  // // Matlab read/write
  // kern->writeMatlabFile("crap.mat", "writtenKern");
  // kern2->readMatlabFile("crap.mat", "writtenKern");
  // if(kern->equals(*kern2))
  //   cout << "MATLAB written " << kern->getName() << " matches read in kernel. Read and write to MATLAB passes." << endl;
  // else
  // {
  //   cout << "FAILURE: MATLAB read in " << kern->getName() << " does not match written out kernel." << endl;
  //   cout << "Matlab written:" << endl;
  //   cout << kern->display(cout) << endl;
  //   cout << "Read back in:" << endl;
  //   cout << kern2->display(cout) << endl;
  //   fail++;
  // }

  // Write to stream.
  // kern->toFile("crap_kern");
  // kern2->fromFile("crap_kern");
  // if(kern->equals(*kern2))
  //   cout << "Text written " << kern->getName() << " matches read in kernel. Read and write to stream passes." << endl;
  // else
  // {
  //   cout << "FAILURE: Text read in " << kern->getName() << " does not match written kernel." << endl;
  //   cout << "Text written:" << endl;
  //   cout << kern->display(cout) << endl;
  //   cout << "Read back in:" << endl;
  //   cout << kern2->display(cout) << endl;
  //   fail++;
  // }
  cout << endl;
  return fail;
}

// test recursive naming structure for kernels
int testKernNaming() {
  int fail = 0;

  // basic kernel
  string var_name = "variance";
  string ls_name = "lengthScale";

  CMatern32Kern matern32;
  matern32.setParamByName(var_name, 1.0);
  matern32.setParamByName(ls_name, 2.0);

  assert(matern32.getParamByName(var_name) == 1.0);
  assert(matern32.getParamByName(ls_name) == 2.0);

  CMatrix ls_bounds(1, 2);
  ls_bounds(0,0) = 0.1;
  ls_bounds(0,1) = 0.5;
  CMatrix var_bounds(1, 2);
  var_bounds(0,0) = 0.3;
  var_bounds(0,1) = 0.7;
  matern32.setBoundsByName(var_name, var_bounds);
  matern32.setBoundsByName(ls_name, ls_bounds);

  assert(var_bounds.equals(matern32.getBoundsByName(var_name)));
  assert(ls_bounds.equals(matern32.getBoundsByName(ls_name)));

  // kernel with one layer of nesting
  var_name = string("matern32_0:variance");
  ls_name = string("matern32_0:lengthScale");
  string white_name = "white_1:variance";
  CWhiteKern white;

  CCmpndKern cmpd_kern;
  cmpd_kern.addKern(&matern32);
  cmpd_kern.addKern(&white);
  cmpd_kern.setParamByName(var_name, 3.0);
  cmpd_kern.setParamByName(ls_name, 4.0);
  cmpd_kern.setParamByName(white_name, 5.0);

  assert(cmpd_kern.getParamByName(var_name) == 3.0);
  assert(cmpd_kern.getParamByName(ls_name) == 4.0);
  assert(cmpd_kern.getParamByName(white_name) == 5.0);

  ls_bounds(0,0) = 1.1;
  ls_bounds(0,1) = 1.5;
  var_bounds(0,0) = 1.3;
  var_bounds(0,1) = 1.7;
  CMatrix white_bounds(1, 2);
  white_bounds(0, 0) = 1.2;
  white_bounds(0, 1) = 1.8;

  cmpd_kern.setBoundsByName(var_name, var_bounds);
  cmpd_kern.setBoundsByName(ls_name, ls_bounds);
  cmpd_kern.setBoundsByName(white_name, white_bounds);

  assert(var_bounds.equals(cmpd_kern.getBoundsByName(var_name)));
  assert(ls_bounds.equals(cmpd_kern.getBoundsByName(ls_name)));
  assert(white_bounds.equals(cmpd_kern.getBoundsByName(white_name)));

  // kernel with two layers of nesting
  CCmpndKern cmpd2;
  cmpd2.addKern(&cmpd_kern);
  var_name = string("cmpnd_0:matern32_0:variance");
  ls_name = string("cmpnd_0:matern32_0:lengthScale");
  white_name = string("cmpnd_0:white_1:variance");

  cmpd2.setParamByName(var_name, 6.0);
  cmpd2.setParamByName(ls_name, 7.0);
  cmpd2.setParamByName(white_name, 8.0);

  assert(cmpd2.getParamByName(var_name) == 6.0);
  assert(cmpd2.getParamByName(ls_name) == 7.0);
  assert(cmpd2.getParamByName(white_name) == 8.0);

  ls_bounds(0,0) = 2.1;
  ls_bounds(0,1) = 2.5;
  var_bounds(0,0) = 2.3;
  var_bounds(0,1) = 2.7;
  white_bounds(0, 0) = 2.2;
  white_bounds(0, 1) = 2.8;

  cmpd2.setBoundsByName(var_name, var_bounds);
  cmpd2.setBoundsByName(ls_name, ls_bounds);
  cmpd2.setBoundsByName(white_name, white_bounds);

  assert(var_bounds.equals(cmpd2.getBoundsByName(var_name)));
  assert(ls_bounds.equals(cmpd2.getBoundsByName(ls_name)));
  assert(white_bounds.equals(cmpd2.getBoundsByName(white_name)));

  // test display of kernel structure
  json sol = cmpd2.display_structure();
  
  json ref;
  ref[var_name]["bounds"] = vector<double>(var_bounds.getVals(), var_bounds.getVals() + 2);
  ref[ls_name]["bounds"] = vector<double>(ls_bounds.getVals(), ls_bounds.getVals() + 2);;
  ref[white_name]["bounds"] = vector<double>(white_bounds.getVals(), white_bounds.getVals() + 2);;
  assert(sol == ref);

  return fail;
}


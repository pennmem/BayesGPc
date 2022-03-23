#include "COptimisable.h"

const double COptimisable::phi=1.618033988749895;
const double COptimisable::cphi=-0.6180339887498949;
const double COptimisable::smallNum=1e-11;



void COptimisable::checkGradients()
{
  double change = ndlutil::GRADCHANGE;
  double origParam = 0.0;
  double objectivePlus = 0.0;
  double objectiveMinus = 0.0;
  int nParams = getOptNumParams();
  
  CMatrix analyticGrad(1, nParams);
  CMatrix numericalDiff(1, nParams);
  CMatrix diffNumericalAnalytic(1, nParams);
  CMatrix params(1, nParams);
  CMatrix origParams(1, nParams);
  
  getOptParams(params);
  origParams.deepCopy(params);
  computeObjectiveGradParams(analyticGrad);
  for(int j=0; j<nParams; j++) 
  {
    origParam = origParams.getVal(j);
    params.setVal(origParam + change, j);
    setOptParams(params);
    objectivePlus = computeObjectiveVal();
    params.setVal(origParam - change, j);
    setOptParams(params);
    objectiveMinus = computeObjectiveVal();
    numericalDiff.setVal(0.5*(objectivePlus - objectiveMinus)/change, j);
    params.setVal(origParams.getVal(j), j);
  }
  
  cout << "Numerical differences:" << endl << numericalDiff << endl;
  cout << "Analytic gradients:" << endl << analyticGrad << endl;
  diffNumericalAnalytic.deepCopy(analyticGrad);
  diffNumericalAnalytic-=numericalDiff;
  cout << "Differences: " << endl << diffNumericalAnalytic << endl;
}

void COptimisable::gdOptimise()
{
  if(getVerbosity()>2)
  {
    cout << "Gradient Descent Optimisation." << endl;
  }
  int nParams = getOptNumParams();
  double objectiveVal = 0.0;
  double oldObjective = 0.0;
  double diffObjective = 0.0;
  double diffParam = 0.0;
  CMatrix params(1, nParams);
  CMatrix oldParams(1, nParams);
  CMatrix gradParams(1, nParams);
  CMatrix changeParams(1, nParams);
  changeParams.zeros();
  getOptParams(params);
  if(evalFunc)
    objectiveVal = computeObjectiveVal();
  for(iter=0; iter<getMaxIters(); iter++)     
  {
    oldParams.deepCopy(params);
    computeObjectiveGradParams(gradParams);
    if(momentum>0)
	{
	  changeParams.axpy(gradParams, -learnRate/momentum);
	  params.axpy(changeParams, momentum);
	  changeParams.scale(momentum);
	}
      else
	{
	  params.axpy(gradParams, -learnRate);	    
	}
      setOptParams(params);
      if(evalFunc)
	{
	  oldObjective = objectiveVal;
	  objectiveVal = computeObjectiveVal();
	  diffObjective = fabs(objectiveVal-oldObjective);
	}
      if(getVerbosity()>2)
	{
	  cout << "Iteration: " << iter << ", objective function: " << objectiveVal << endl;
	}
      diffParam = params.maxAbsDiff(oldParams);
      if(diffObjective<objectiveTol && diffParam<parameterTol)
	{
	  cout << "Param difference: " << diffParam << endl;
	  cout << "Objective difference: " << diffObjective << endl;
	  cout << "Converged .." << endl;
	  break;
	}
      
    } 
  cout << "Parameters: " << endl;
  cout << params << endl;
  /*       if(iter>=getMaxIters()) */
  /* 	cout << "Maximum iterations exceed in gdOptimise(), objective change, " << diffObjective << ", parameter change, " << diffParam << endl; */
}
void COptimisable::gdPullbackOptimise()
{
  if(getVerbosity()>2)
  {
    cout << "Gradient Descent with pullback Optimisation." << endl;
  }


  int nParams = getOptNumParams();
  double objectiveVal = 0.0;
  double oldObjective = 0.0;
  double diffObjective = 0.0;
  double diffParam = 0.0;
  CMatrix params(1, nParams);
  CMatrix oldParams(1, nParams);
  CMatrix gradParams(1, nParams);
  CMatrix changeParams(1, nParams);
  changeParams.zeros();
  getOptParams(params);
  objectiveVal = computeObjectiveVal();
  for(iter=0; iter<getMaxIters(); iter++)     
  {
    while(1)
    {
      oldParams.deepCopy(params);
      computeObjectiveGradParams(gradParams);
      params.axpy(gradParams, -learnRate);	    
      setOptParams(params);
      oldObjective = objectiveVal;
      objectiveVal = computeObjectiveVal();
      diffObjective = oldObjective-objectiveVal;
      if(diffObjective<0)
      {
	params.deepCopy(oldParams);
	learnRate = learnRate/2;
	objectiveVal=oldObjective;
	setOptParams(params);
	if(getVerbosity()>2)
	  cout << "Back tracking, learning rate: " << learnRate << endl;
      }
      else
      {
	learnRate = learnRate*1.1;
	break;
      }
    }
    if(getVerbosity()>2)
    {
      cout << "Iteration: " << iter << ", objective function: " << objectiveVal << endl;
    }
    diffParam = params.maxAbsDiff(oldParams);
    if(diffObjective<objectiveTol && diffParam<parameterTol)
    {
      cout << "Param difference: " << diffParam << endl;
      cout << "Objective difference: " << diffObjective << endl;
      cout << "Converged .." << endl;
      break;
    }
    
  } 
  cout << "Parameters: " << endl;
  cout << params << endl;
  /*       if(iter>=getMaxIters()) */
  /* 	cout << "Maximum iterations exceed in gdOptimise(), objective change, " << diffObjective << ", parameter change, " << diffParam << endl; */
}
double COptimisable::oneDObjectiveVal(const double val)
{
  DIMENSIONMATCH(paramStoreOne.getRows()==1);
  DIMENSIONMATCH(paramStoreTwo.getRows()==1);
  DIMENSIONMATCH(paramStoreOne.getCols()==getOptNumParams());
  DIMENSIONMATCH(paramStoreTwo.getCols()==getOptNumParams());
  getOptParams(paramStoreOne);
  paramStoreTwo.deepCopy(paramStoreOne);
  // add val times direction to the parameters.
  paramStoreTwo.axpy(direction, val);
  setOptParams(paramStoreTwo);
  double objective = computeObjectiveVal();
  setOptParams(paramStoreOne);
  return objective;
}
void COptimisable::lbfgs_b_Optimise()
{
  assert(bounds_set);
  if(getVerbosity()>2)
  {
    cout << "Limited Memory Bounded BFGS Optimisation." << endl;
  }

  int maxiters = 50;
  int funcEval = 0;
  int maxFuncEval = 10000;
  
  int nParams = getOptNumParams();
  int memSize = 10;  // number of corrections
  double factr = 1e7; // solution accuracy based on function values
  double pgtol = 1e-7; // solution accuracy based on gradient values
  int maxls = 20;  // max number of line search iterations
  
  int iprint;  // verbosity
  if (getVerbosity() < 2) {iprint = -1;}
  else if (getVerbosity() <= 3) {iprint = 0;}  // print output info only on the last iteration
  else if (getVerbosity() <= 5) {iprint = 5;}  // print output info every 5 iterations
  else if (getVerbosity() > 5) {iprint = 101;}  // print output info on every iteration

  // track best objective across iterations (to guard against numerical errors) and random restarts
  double f_best = INFINITY;
  double f = 0.0; // function value, not modified within lbgsb

  // current restart
  int restart = -1;
  // convergence flag
  bool conv = false;

  CMatrix X(1, nParams);
  getOptParams(X);
  // tracks current best x across iterations (in case of numerical errors) and random restarts
  CMatrix X_best(X);
  // tracks whether a single successful update has occurred
  // e.g. whether CGp can update at all with the newest sample
  CMatrix X_orig(X);
  CMatrix g(1, nParams);

  while (restart < n_restarts && !conv) {
    iter = 0;
    restart++;

    // LBFGS-B settings that must be reset across random restarts
    double* Xvals = new double[nParams];
    double* gvals = new double[nParams]; // gradient, not clear whether this is modified within lbgsb
    int* nbd = new int[nParams];
    for (int i = 0; i < nParams; i++) {
      nbd[i] = 2;
    }
    double* work = new double[nParams*(2*memSize + 5) + 12*(memSize + 1)*memSize];  // (2mmax + 5)nmax + 12mmax^2 + 12mmax
    int* iwa = new int[3*nParams];  // length 3nmax
    char* task = new char[61];  // length 60, add one character since FORTRAN doesn't null-terminate
    char* csave = new char[61]; // message, add one character since FORTRAN doesn't null-terminate
    for (int i = 0; i < 60; i++) {
      task[i] = 'a';  // ' '
      csave[i] = 'a'; // ' '
    }
    task[60] = '\0';
    csave[60] = '\0';
    strcpy(task, "START");
    task[5] = ' ';
    bool* lsave = new bool[4]; // flags for controlling constraints and other algorithm settings
    int* isave = new int[44]; // integer array of length 44, contains diagnostic info
    double* dsave = new double[29]; // working array, length 29

    rip out bfgs, which is giving fortran compile error with 64-bit lapack?
        decent amount of work, might not end up working,
        would be undesirable for future compatability, would remove functionality that is working in Unix-based systems
    or need to replace lapack subroutines (compiled for 64-bit integers) with subroutines
            inside ndlfortran_lbfgs.f that could then be compiled with 32-bit...
            or could try compiling just LBFGS_B with 64-bit, everything else with 32-bit


    // LBFGS-B search run
    try {
      while(true)
      {
        f = computeObjectiveGradParams(g);
        X.toArray(Xvals);
        g.toArray(gvals);
        // store best x across iterations and random restarts to guard against exceptions (in 
        // particular, due to numerical instability)
        if (f < f_best) {
          f_best = f;
          X_best = X;
        }
        cout << "Before lbfgs_b" << endl;
//        cout << "nParams" << nParams << endl;
//        cout << "memSize" << memSize << endl;
//        cout << "nbd " << endl; // const C2F_BASE_INT_TYPE*
//        for (int i = 0; i < nParams; i++) { cout << nbd[i] << " "; }
//        cout << endl;

//        cout << "iwa" << endl; // C2F_BASE_INT_TYPE*
//        for (int i = 0; i < 3*nParams; i++) { iwa[i] = i; cout << iwa[i] << " "; }
//        cout << endl;
//        cout << "f" << f << endl; // constt double&
//        cout << "g" << g << endl; // double array, const double*
//        cout <<"factr" << factr << endl; // const double&
//        cout << "pgtol" << pgtol << endl; // const double&

//        cout << "dsave" << endl; // double*
//        for (int i = 0; i < 29; i++) { cout << dsave[i] << " "; }
//        cout << endl;

//        cout << "csave" << endl;
//        for (int i = 0; i < 60; i++) { cout << csave[i] << " "; }
//        cout << endl;

        cout << "lsave" << endl;
        for (int i = 0; i < 4; i++) { cout << lsave[i] << " "; }
        cout << endl;

        setulb_( // pass arrays to fortran with pointers
          nParams,
          memSize,  // number of corrections
          Xvals, // length n
          lower_bounds.getVals(), // length n
          upper_bounds.getVals(), // length n
          nbd, // length n, integer flags coding bounds (0: none, 1: lower, 2: both, 3: upper) for each variable
          f, // function value, not clear whether this is modified within lbgsb
          gvals, // gradient, not clear whether this is modified within lbgsb
          factr, // solution accuracy based on function values
          pgtol, // solution accuracy based on gradient values
          work, // length (2mmax + 5)nmax + 12mmax^2 + 12mmax
          iwa, // length 3nmax
          task,  // length 60
          iprint,  // verbosity
          csave, // message
          lsave, // flags for controlling constraints and other algorithm settings
          isave, // integer array of length 44, contains diagnostic info
          dsave, // working array, length 29
          C2F_INT_PTR_CONVERSION maxls  // max number of line search iterations
        );
        X.fromArray(Xvals);
        setOptParams(X);
        if (getVerbosity() >= 2) { cout << "task " << task << endl; }
        
        if (strncmp(task, "CONV", 4) == 0) {
          break;
        }
        else if (strncmp(task, "FG", 2) == 0) {
          // reevaluate objective and gradient
          funcEval++;
        }
        else if (strncmp(task, "NEW_X", 5) == 0) {
          // new iteration
          iter++;

          if (iter >= maxIters) {
            memset(task, ' ', 60);
            strcpy(task, "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT");
          }
          else if (funcEval >= maxFuncEval) {
            memset(task, ' ', 60);
            strcpy(task, "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT");
          }
        }
        else if (strncmp(task, "ABNORMAL_TERMINATION_IN_LNSRCH", 30) == 0) {
          // does not appear to be an error state, though it is not clear what exactly this message indicates from the FORTRAN code
          // indicates failure to descend after 20 line search iterations, potentially caused by
          // 1. gradient computations not matching function 
          // 2. numerical error dominating the computation
          cout << endl << task << endl << "Numbers of iterations: " << iter << endl;
          cout << "log params" << endl << X << endl;
          cout << "grads" << endl << g << endl << endl;
          iter++;
          break;
        }
        else {  // other error/warning state
          cout << "Unknown LBFGS-B warning/error: " << endl << task << endl;
          break;
        }
      }
    }
    catch (const ndlexceptions::MatrixError& e) {  // assume error of numerical stability or non-PSD
      cout << "Matrix error in LBFGS-B: " << endl;
      cout << e.what() << endl;
      if (restart == n_restarts - 1) {
        if (X_best.equals(X_orig)) {
          cout << "LBFGS-B: Failure to successfully fit new X_best after " << n_restarts << " random restarts. Throwing exception." << endl;
          throw e;
        }
        else { setOptParams(X_best); }
      }
      else {
        cout << "Restarting optimization with random X (restart " << restart << ")" << endl;
        boost::random::uniform_real_distribution<> dist(0.0, 1.0);
        boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > gen(rng, dist);
        for (unsigned int i = 0; i < nParams; i++) {
            X.setVal(bounds.getVal(i, 0) + gen() * (bounds.getVal(i, 1) - bounds.getVal(i, 0)), i);
        }
      }
    }
    catch (const std::exception& e) {
      cout << "Unhandled exception in LBFGS-B: " << endl;
      cout << e.what() << endl;
      cout << "Exiting search process." << endl;
      throw e;
    }

    if (strncmp(task, "CONV", 4) == 0) {
      setOptParams(X_best);
      conv = true;
      if (getVerbosity() >= 2) { cout << "LBFGS-B successfully converged." << endl; }
    }
    else if (iter >= maxIters || funcEval >= maxFuncEval) {
      if (getVerbosity() >= 0) { cout << "COptimisable: lbfgs_b message: " << task << endl; }
      setOptParams(X_best);
      conv = true;
    }
    else if (strncmp(task, "ABNORMAL_TERMINATION_IN_LNSRCH", 30) == 0) {
      if (getVerbosity() >= 0) { cout << "COptimisable: lbfgs_b message: " << task << endl; }
      setOptParams(X_best);
      conv = true;
    }
    // pass through if errors occurred in function evaluation before LBFGS-B search run once
    else if (strncmp(task, "START", 5) == 0) {}
    else {
      cout << "COptimisable lbfgs_b error: " << endl << task << endl;
      if (restart == n_restarts - 1) {
        if (X_best.equals(X_orig)) {
          cout << "LBFGS-B: Failure to successfully fit new X_best after " << n_restarts << " random restarts. Throwing exception." << endl;
          throw ndlexceptions::Error(string("Unhandled error in lbfgsb:   ") + string(task));
        }
        else { setOptParams(X_best); }
      }
      else {
        cout << "Restarting optimization with random X (restart " << restart << ")" << endl;
        boost::random::uniform_real_distribution<> dist(0.0, 1.0);
        boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > gen(rng, dist);
        for (unsigned int i = 0; i < nParams; i++) {
            X.setVal(bounds.getVal(i, 0) + gen() * (bounds.getVal(i, 1) - bounds.getVal(i, 0)), i);
        }
      }
    }
    delete[] Xvals;
    delete[] nbd;
    delete[] gvals;
    delete[] work;
    delete[] iwa;
    delete[] task;
    delete[] csave;
    delete[] lsave;
    delete[] isave;
    delete[] dsave;
  }

  if (max(X) > log(1e10)) {
    cout << "Warning: large log parameters: " << endl << X << endl;
    cout << "gradients: " << endl << g << endl;
  }
}
void COptimisable::lbfgsOptimise()
{
  if(getVerbosity()>2)
  {
    cout << "Limited Memory BFGS Optimisation." << endl;
  }
  int nParams = getOptNumParams();
  int iflag = 0;
  int memSize = 10;
  double* Xvals = new double[nParams];
  double* work = new double[nParams*(2*memSize+1) + 2*memSize];
  double* gvals = new double[nParams];
  double* diagVals = new double[nParams];
  
  CMatrix X(1, nParams);
  CMatrix g(1, nParams);
  int iPrint[2] ={-1, 0};
  if(getVerbosity()>2)
  {
    iPrint[0] = 1;
  }
  double f = 0.0;
  getOptParams(X);
  while(true)
  {
    f = computeObjectiveGradParams(g);
    X.toArray(Xvals);
    g.toArray(gvals);
    lbfgs_(nParams, memSize, Xvals, f, gvals, 0, diagVals, iPrint, getObjectiveTol(), getParamTol(), work, iflag);
    if(iflag<=0)
    {
      if(iflag==-1)
      {
        cout << "Warning: lbfgsOptimise: linesearch failed." << endl;
        break;
      }
      else if(iflag == -2)
      {
      	throw ndlexceptions::Error("An element of the inverse Hessian provided is not positive.");
      }
      else if(iflag == -3)
      {
	      throw ndlexceptions::Error("Inproper input to lbfgs_.");
      }
    }
    else if(iflag==0)
    {
      break;
    }
    else if(iflag==1)
    {
      X.fromArray(Xvals);
      setOptParams(X);
      cout << "log params" << endl << X << endl;
      cout << "grads" << endl << g << endl;
      funcEval++;
    }
    else
    {
      throw ndlexceptions::Error("Unhandled iflag.");
    }
  }
  if (max(X) > log(1e10)) {
    cout << "Warning: log parameters massive: " << endl << X << endl;
    cout << "Gradients: " << endl << g << endl;
  }
}
void COptimisable::scgOptimise()
{
  // taken from the paper by Martin Moller: "A scaled conjugate gradient algorithm for fast supervised learning".
  if(getVerbosity()>2)
  {
    cout << "Scaled Conjugate Gradient Optimisation." << endl;
  }
  int nParams = getOptNumParams();
  //double objectiveVal = 0.0;
  //double oldObjective = 0.0;
  //double diffObjective = 0.0;
  //double diffParam = 0.0;
  
  bool success = true;
  double beta;
  double Delta;
  double lambda;
  double lambdaBar;
  double sigma;
  //double rr;

  CMatrix w(1, nParams);
  CMatrix wPlus(1, nParams);

  CMatrix r(1, nParams);
  CMatrix p(1, nParams);
  CMatrix rp(1, nParams);
  CMatrix s(1, nParams);
  getOptParams(w);

  const double m_step = 1.0e-4;
  const double m_reg = 1.0;
  double oldObj = 0.0;
  //double obj = oldObj;
  double mu = 0.0;
  //double theta = 0.0;
  double sigmaInv = 0.0;
  double delta = 0.0;
  double alpha = 0.0;
  double newObj = 0.0;
  //double gamma = 0.0;
  //int j = 1;					// j counts number of iterations.
  
  // 1
  lambda = m_reg; // lambda is the scale??
  lambdaBar = 0.0;
  oldObj = computeObjectiveGradParams(r);
  r.negate();
  p.deepCopy(r);

  for(iter=1; iter<=getMaxIters(); iter++)
  {
    // should check that p is not infinite above
    double normp = p.normRow(0);
    double normp2 = normp*normp;
    // 2
    if(success)
	{
	  // can get a divide by zero here if pp is too small.
	  
	  sigma = m_step/normp;
	  wPlus.deepCopy(w);
	  wPlus.axpy(p, sigma);
	  setOptParams(wPlus);
	  computeObjectiveGradParams(s);
	  sigmaInv = 1/sigma;
	  s.scale(sigmaInv);
	  s.axpy(r, sigmaInv);
	  delta = s.dotRowRow(0, p, 0);
	}
      
    // 3 Scale s_iter
    double lambdaDiff = lambda-lambdaBar;
    s.axpy(p, lambdaDiff);
    delta += lambdaDiff*normp;
      
    // 4 
    if(delta <= 0.0) // Make Hessian positive definite.
	{
	  double deltaOverNormp2 = delta/normp2;
	  s.axpy(p, (lambda-2.0*deltaOverNormp2));
	  
	  lambdaBar = 2.0*(lambda - deltaOverNormp2);
	  delta = lambda*normp2 - delta; 
	  lambda = lambdaBar;
	}
      
    // 5  Calculate step size.
    mu=p.dotRowRow(0, r, 0);
    alpha = mu/delta;
      
    // 6 Compute the comparison parameter.
    wPlus.deepCopy(w);
    wPlus.axpy(p, alpha);
    setOptParams(wPlus);
    newObj = computeObjectiveVal();
    Delta = 2.0*delta*(oldObj - newObj)/(mu*mu);

    // 7 Check whether a successful error reduction can be made.
    if(Delta >= 0.0)  // update is successful
      {
	w.deepCopy(wPlus); 	  
	oldObj = newObj;
	computeObjectiveGradParams(rp);
	rp.negate();
	lambdaBar = 0; 
	success = true;
	  // 7.a Check for algorithm restart.
	  if(iter % nParams == 0) // restart algorithm
	    p.deepCopy(rp); 
	  else
	    {	      
	      double rpnorm2 = rp.norm2Row(0);
	      double rrp = r.dotRowRow(0, rp, 0); 
	      
	      beta = (rpnorm2 - rrp)/mu;
	      p.scale(beta); 
	      p.axpy(rp, 1.0);
	    }
	  
	  r.deepCopy(rp);
	  
	  // 7.b Reduce the scale parameter
	  if(Delta >= 0.75) lambda *= 0.5;
	  if(lambda<1e-15) lambda = 1e-15;
	}
    else // no reduction in error is possible.
      {
	setOptParams(w);
	lambdaBar = lambda; 
	success = false; 
      }
    
    // 8 Increase the scale parameter
    if(Delta < 0.25) lambda *= 4.0;
    
    // 9 Check for convergence       
    if(getVerbosity()>2)
      cout << "Iteration: " << iter << " Error: " << oldObj << " Scale: " << lambda << endl;
    if (success && fabs(p.max()*alpha) < getParamTol() && max(fabs(newObj-oldObj)) < getObjectiveTol())
      {
	if(getVerbosity()>2)
	{
	  cout << "Convergence criterion for parameters and objective met" << endl;
	  cout << "Largest tolerance " << fabs(newObj - oldObj) << endl;
	}
	return;
      }
  }
  cout << "Warning: Maximum number of iterations has been exceeded" << endl;
}
void COptimisable::cgOptimise()
{
  if(getVerbosity()>2)
  {
    cout << "Conjugate Gradient Optimisation." << endl;
  }
  
  // a C++ translation of Carl Rasmussen's minimize function
  int nParams = getOptNumParams();
  bool success = true;
  
  const double INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket.
  const double EXT = 3.0; // extrapolate maximum 3.0 times the current step-size.
  const unsigned int MAX = 20; // maximum 20 function evaluations per line search
  const double RATIO = 10.0; // maximum allowed slope ratio.
  const double SIG = 0.1; 
  const double RHO = SIG/2.0; // SIG and RHO are the constants controlling the Wolfe-Powell conditions. 
  
  
  CMatrix df0(1, nParams); // gradient direction.
  CMatrix dF0(1, nParams); // gradient direction.
  CMatrix s(1, nParams); // search direction.
  CMatrix X(1, nParams); // parameter vector.
  CMatrix X0(1, nParams); // parameter vector.
  CMatrix wPlus(1, nParams); // parameter vector.
  
  double red = 1.0; 
  iter = 0;  // start with zero iterations.
  funcEval = 0; // start with zero function evaluations.
  bool ls_failed = false;  // no previous line search has failed.
  
  // compute initial gradient and function value.
  double f0 = computeObjectiveGradParams(df0);
  funcEval++;  // add to functional computation tally.
  s.deepCopy(df0);
  s.negate(); // initial search direction (steepest descent)
  double d0 = -s.norm2Row(0);
  
  getOptParams(X);
  double F0 = 0.0;
  double x1 = 0.0;
  double x2 = 0.0;
  double x3 = red/(1-d0); // initial step size is red/(|s|+1)
  double x4 = 0.0;
  double f1 = 0.0;
  double f2 = 0.0;
  double f3 = 0.0;
  double f4 = 0.0;
  double d1 = 0.0;
  double d2 = 0.0;
  double d3 = 0.0;
  double d4 = 0.0;
  double A = 0.0;
  double B = 0.0;
  vector<double> fX;
  
  CMatrix df3(1, nParams);
  int M = 0;	 
  while((isIterTerminate() 
	 && iter<getMaxIters())  
	|| (isFuncEvalTerminate() 
	    && funcEval<getMaxFuncEvals()))
  {
    iter++; //update number if iterations
    // make a copy of current values
    X0.deepCopy(X);
    F0 = f0;
    dF0.deepCopy(df0);
    if(MAX <= getMaxFuncEvals() || isFuncEvalTerminate())
    {
      M = MAX;
    }
    else 
    {
      M = getMaxFuncEvals();
    }
    while(true) // keep doing line search until break.
    {
      x2 = 0.0;
      f2 = f0;
      d2 = d0;
      f3 = f0;
      df3.deepCopy(df0);
      success = false;
      while(!success && M>0)
      {
	try
	{
	  M--;
	  funcEval++;
	  // compute gradient
	  wPlus.deepCopy(X);
	  wPlus.axpy(s, x3);
	  setOptParams(wPlus);  
	  f3 = computeObjectiveGradParams(df3);
	  if(!(isnan(f3) || isinf(f3) || df3.isAnyNan() || df3.isAnyInf()))
	  {
	    // clean computation of gradients, success!
	    success = true;
	  }
	  else
	  {
	    if(getVerbosity()>1)
	      cout << "cgOptimise: Warning gradient or function value was NaN or inf." << endl;
	  }
	}
	catch(ndlexceptions::MatrixNonPosDef err)
	{
	  if(getVerbosity()>1)
	    cout << "cgOptimise: Matrix non-positive definite in gradient of function value computation." << endl;
	}
	catch(ndlexceptions::MatrixConditionError err)
	{
	  if(getVerbosity()>1)
	    cout << "cgOptimise: Matrix conditioning error in gradient of function value computation." << endl;
	}
	catch(ndlexceptions::MatrixSingular err)
	{
	  if(getVerbosity()>1)
	    cout << "cgOptimise: Matrix singularity error in gradient of function value computation." << endl;
	}
	if(!success) // if any of these errors have occured, pull back and retry.
	{
	  cout << "Pulling back by half." << endl;
	  x3 = (x2 + x3)/2; 
	}
      }
      if(f3<F0) // keep best values.
      {
	X0.deepCopy(X);
	X0.axpy(s, x3);
	F0 = f3;
	dF0.deepCopy(df3);
      }
      d3 = df3.dotRowRow(0, s, 0);
      if(d3>SIG*d0 || f3>f0+x3*RHO*d0 || M==0) // Is line search over?
      {
	break;
      }
      x1 = x2; f1 = f2; d1 = d2;   // move point 2 to point 1.
      x2 = x3; f2 = f3; d2 = d3;   // move point 3 to point 2.
      A = 6.0*(f1-f2)+3.0*(d2+d1)*(x2-x1); // do cubic extrapolation.
      B = 3.0*(f2-f1)-(2.0*d1+d2)*(x2-x1);
      x3 = x1-d1*((x2-x1)*(x2-x1))/(B + sqrt(B*B-A*d1*(x2-x1)));  // new extrapolation point.
      // checks on new extropolation point.
      if(isnan(x3) || isinf(x3) || x3 < 0.0)
	x3=x2*EXT;   // extrapolate maximum ammount.
      else if(x3>x2*EXT) // new point beyond extrapolation limit?
	x3=x2*EXT;   // extrapolate maximum ammount.
      else if(x3<x2+INT*(x2-x1))  // new point too close to previous point?
	x3=x2+INT*(x2-x1);  // extrapolate minimum amount
    }
    while((abs(d3)>-SIG*d0 || f3>f0+x3*RHO*d0) && M>0)  // keep interpolating.
    {
      if(d3>0 || f3>f0+x3*RHO*d0) // choose subinterval.
      {
	x4=x3; f4=f3; d4=d3;   // move point 3 to point 4.
      }
      else
      {
	x2=x3; f2=f3; d2=d3;  // move point 3 to point 2.
      }
      if(f4>f0)
      {
	x3 = x2-(0.5*d2*((x4-x2)*(x4-x2)))/(f4-f2-d2*(x4-x2));  // quadratic interpolation.
      }
      else
      {
	A = 6.0*(f2-f4)/(x4-x2)+3*(d4+d2);   // cubic interpolation.
	B = 3.0*(f4-f2)-(2*d2+d4)*(x4-x2);
	x3 = x2+(sqrt(B*B-A*d2*(x4-x2)*(x4-x2))-B)/A;
      }
      if(isnan(x3) || isinf(x3))
	x3 = (x2+x4)/2.0;                // bisect if there was a numerical problem.
      x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2)); // don't accept too close.
      // add s times x3 to wPlus.
      wPlus.deepCopy(X);
      wPlus.axpy(s, x3);
      setOptParams(wPlus);  
      f3 = computeObjectiveGradParams(df3);
      if(f3<F0) // keep best values.
      {
	F0 = f3;
	dF0.deepCopy(df3);
	X0.deepCopy(wPlus);
      }
      funcEval++;
      M--;
      d3 = df3.dotRowRow(0, s, 0);  // new slope.
    } // end of interpolation.
    if (abs(d3)<-SIG*d0 && f3 < f0+x3*RHO*d0)  // if line search succeeded.
    {
      X.deepCopy(wPlus);
      f0 = f3;
      fX.push_back(f0);
      if(getVerbosity()>2)
        cout << "Iteration: " << iter << " Error: " << f0  << endl;
      // Polack-Ribiere CG direction.
      s.scale((df3.norm2Row(0) - df0.dotRowRow(0, df3, 0))/df0.norm2Row(0));
      s.axpy(df3, -1.0); 
      df0.deepCopy(df3); // swap derivatives
      d3=d0; d0=df0.dotRowRow(0, s, 0);
      if(d0>0)
      {
        // not negative --- use steepest descent.	
	s.deepCopy(df0);
	s.negate(); 
	d0 = -s.norm2Row(0);
      }
      x3 = x3* min(RATIO, d3/(d0-__DBL_MIN__));
      ls_failed = false;
    }
    else
    {
      // revert the line search failed, restore best point.
      X.deepCopy(X0);
      f0=F0;
      df0.deepCopy(dF0);
      if(ls_failed || (isIterTerminate() && iter>=getMaxIters()) || (isFuncEvalTerminate() && funcEval>=getMaxFuncEvals()))
      { 
	// line search failed twice in a row, or iterations are exceeded.
	//Need a tolerance check here!!.
	break;
      }
      // restart from steepest descent direction.
      s.deepCopy(df0);
      s.negate();
      d0 = -s.norm2Row(0);
      x3 = 1/(1-d0);
      ls_failed = true;
    }
  }
  if(isIterTerminate() && iter >= getMaxIters())
  {
    // max iters exceeded.
    cout << "cgOptimise: Warning: Maximum number of iterations has been exceeded" << endl;
  }
  if(isFuncEvalTerminate() && funcEval >= getMaxFuncEvals())
  {
    // max func evaluations exceeded.
    cout << "cgOptimise: Warning: Maximum number of function evalutaions has been exceeded" << endl;
  }
  
}

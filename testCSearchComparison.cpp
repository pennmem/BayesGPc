#include "CSearchComparison.h"
using namespace std;

int testType(const string Type);
int testComparison(const string Type, const string fileName);

int main()
{
  int fail=0;
  try
  {
    fail += testType("welch_anova");
    fail += testType("welch_ttest");

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

int testType(const string Type)
{
  string fileName = "np_files" + ndlstrutil::dirSep() + "testCSearchComparison_" + Type + ".npz";

  if (Type.compare("welch_anova") != 0 && Type.compare("welch_ttest") != 0)
  {
    throw ndlexceptions::Error("Unrecognised test requested.");
  }
  int fail = testComparison(Type, fileName);
  return fail;
}
int testComparison(const string Type, const string fileName)
{
  int fail = 0;
  if (Type.compare("welch_anova") == 0) {
    cout << fileName << endl;
    cnpy::npz_t npz = cnpy::npz_load(fileName.c_str());

    double tol = 1e-6;
    cout << "Test absolute max difference tolerance: " << tol << endl;

    int num_test = (int)(*npz["n_tests"].data<double>());

    for (int i = 0; i < num_test; i++) {
      int num_group = (int)(*npz["n_groups"].data<double>());
      vector<double> mus;
      vector<double> sems;
      vector<double> ns;
      for (int j = 0; j < num_group; j++) {
        string key("x" + to_string(j) + "_" + to_string(i));
        CMatrix X(npz[key].data<double>(), npz[key].shape[0], 1);
        mus.push_back(meanCol(X).getVal(0));
        ns.push_back((double)npz[key].shape[0]);
        sems.push_back(stdCol(X, 1).getVal(0)/std::sqrt(ns[j]));
      }

      TestStruct res = anova_welch(mus, sems, ns);
      CMatrix sol(res.pval);
      CMatrix ref(npz["p" + to_string(i)].data<double>(), 1, 1);

      double diff;
      diff = sol.maxRelDiff(ref);
      if(diff < tol)
        cout << Type << " p-value matches within " << diff << " max relative difference." << endl;
      else
      { 
        cout << "FAILURE: " << Type << " p-value." << endl;
        cout << "Maximum absolute difference: " << diff << endl;    
        fail++;
      }

      sol(0) = res.stat;
      ref(0) = *npz["stat" + to_string(i)].data<double>();
      diff = sol.maxRelDiff(ref);
      if(diff < tol)
        cout << Type << " test stat matches within " << diff << " max relative difference." << endl;
      else
      { 
        cout << "FAILURE: " << Type << " test stat." << endl;
        cout << "Maximum relative difference: " << diff << endl;    
        fail++;
      }
    }
  }
  else if  (Type.compare("welch_ttest") == 0) {
    cnpy::npz_t npz = cnpy::npz_load(fileName.c_str());

    double tol = 1e-6;
    cout << "Test absolute max difference tolerance: " << tol << endl;

    int num_test = (int)(*npz["n_tests"].data<double>());

    for (int i = 0; i < num_test; i++) {
      int num_group = (int)(*npz["n_groups"].data<double>());
      vector<double> mus;
      vector<double> sems;
      vector<double> ns;
      for (int j = 0; j < num_group; j++) {
        string key("x" + to_string(j) + "_" + to_string(i));
        CMatrix X(npz[key].data<double>(), npz[key].shape[0], 1);
        mus.push_back(meanCol(X).getVal(0));
        ns.push_back((double)npz[key].shape[0]);
        sems.push_back(stdCol(X, 1).getVal(0)/std::sqrt(ns[j]));
      }

      TestStruct res = ttest_welch(mus[0], mus[1], sems[0], sems[1], ns[0], ns[1]);
      CMatrix sol(res.pval);
      CMatrix ref(npz["p" + to_string(i)].data<double>(), 1, 1);

      double diff;
      diff = sol.maxRelDiff(ref);
      if(diff < tol)
        cout << Type << " p-value matches within " << diff << " max relative difference." << endl;
      else
      { 
        cout << "FAILURE: " << Type << " p-value." << endl;
        cout << "Maximum absolute difference: " << diff << endl;    
        fail++;
      }

      sol(0) = res.stat;
      ref(0) = *npz["stat" + to_string(i)].data<double>();
      diff = sol.maxRelDiff(ref);
      if(diff < tol)
        cout << Type << " test stat matches within " << diff << " max relative difference." << endl;
      else
      { 
        cout << "FAILURE: " << Type << " test stat." << endl;
        cout << "Maximum relative difference: " << diff << endl;    
        fail++;
      }
    }
  }

  return fail;
}


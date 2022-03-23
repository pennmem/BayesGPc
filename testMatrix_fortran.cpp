#include <iostream>
#include <ctime>
#include "ndlexceptions.h"
#include "CMatrix.h"
using namespace std;
const double MAXDIFF=1e-13;

int testCopy();
int testDot();
int testNorm();
int testInv();
int testCholesky();
//int testRandn();
//int testGemm();
//int testSyrk();
//int testTrmm();
//int testTrsm();
//int testGemv();
//int testGer();
//int testSyr();
//int testSymm();
//int testSymv();
//int testSysv();
//int testSyev();
//int testAxpy();
//int testScale();

int main()
{
  int fail = 0;
  time_t seconds;
  time(&seconds);
  ndlutil::init_genrand((unsigned long)seconds);

  try
    {
//      fail += testSyev();
//      cout << "before testInv()" << endl;
      fail += testInv();
      fail += testCholesky();
//      fail += testRandn();

      // Level 3 Blas
//      fail += testGemm();
//      fail += testSyrk();
//      fail += testTrmm();
//      fail += testTrsm();
      // Level 2 Blas
//      fail += testGemv();
//      fail += testSymm();
//      fail += testSymv();
//      fail += testSysv();
//      fail += testGer();
//      fail += testSyr();

      // Level 1 Blas
      fail += testCopy();
//      fail += testAxpy();
//      fail += testScale();
      fail += testDot();
      fail += testNorm();
      cout << endl << "Total number of failures: " << fail << endl;
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

int testCopy()
{
    int fail = 0;
    CMatrix A(2, 2, 0.0);

    CMatrix B(2, 2);
    B.copy(A);
    if(abs(A.maxAbsDiff(B)) < ndlutil::MATCHTOL) {
        cout << "copy (dcopy) matches." << endl;
    }
    else
    {
      cout << "FAILURE: copy (dcopy)." << endl;
      fail++;
    }
    return fail;
}

int testDot()
{
  int fail = 0;
  CMatrix A(1, 5);
  for (int i = 0; i < 5; i++) { A.setVal((double) i, 0, i); }
  CMatrix B(1, 5);
  for (int i = 0; i < 5; i++) { B.setVal((double) i, 0, i); }
  if(abs(30.0 - A.dotRowRow(0, B, 0))<ndlutil::MATCHTOL)
    cout << "dotRowRow matches." << endl;
  else
    {
      cout << "FAILURE: dotRowRow." << endl;
      fail++;
    }
  return fail;
}
int testNorm()
{
  int fail = 0;
  CMatrix A(1, 5);
  for (int i = 0; i < 5; i++) { A.setVal((double) i, 0, i); }
  CMatrix B(1, 5);
  for (int i = 0; i < 5; i++) { B.setVal((double) i, 0, i); }
  double norm = A.norm2Row(0);
  if(abs(30.0-norm)<ndlutil::MATCHTOL)
    cout << "norm2Row matches." << endl;
  else
    {
      cout << "FAILURE: norm2Row." << endl;
      fail++;
    }
  return fail;
}
int testInv()
{
  // implicitly tests dgetrf.
  int fail = 0;
  // Matrix inverse test
  CMatrix A(2, 2, 0.0);
  for (int i = 0; i < 2; i++) { A.setVal((double) 2, i, i); }
  CMatrix Ainv(2, 2, 0.0);
  for (int i = 0; i < 2; i++) { Ainv.setVal((double) 0.5, i, i); }
  cout << A << endl << Ainv << endl;
  A.inv();
  cout << Ainv << endl;
  if(Ainv.equals(A))
    cout << "Matrix inverse matches." << endl;
  else
  {
    cout << "FAILURE: Matrix inverse." << endl;
    fail++;
  }
  return fail;
}
int testCholesky()
{
  // implicitly tests dpotrf
  int fail = 0;
  // Lower Cholesky test.
//  cout << "size of int: " << sizeof(int) << endl;
//  cout << "size of long int: " << sizeof(long int) << endl;
//  cout << "size of long long int: " << sizeof(long long int) << endl;
  CMatrix C(3, 3, 0.0);
  for (int i = 0; i < C.getRows(); i++) { C.setVal((double) 4, i, i); }
  C.setSymmetric(true);
  C.chol("L");

  CMatrix L(3, 3, 0.0);
  for (int i = 0; i < L.getRows(); i++) { L.setVal((double) 2, i, i); }
  if(L.equals(C))
    {
      cout << "Lower Cholesky factor matches."<< endl;
    }
  else
    {
      cout << "FAILURE: Lower Cholesky" << endl;
      fail++;
    }
  return fail;
}
//int testRandn()
//{
//  int fail = 0;
//  int nrows = 1000;
//  int ncols = 10;
//  // Random number test.
//  CMatrix normRand(nrows, ncols);
//  normRand.randn();
//  CMatrix mean = meanCol(meanRow(normRand));
//  if(abs(mean.getVal(0))<1e-2)
//    cout << "randn mean matches." << endl;
//  else
//    {
//      cout << "POSSIBLE FAILURE: randn mean " << mean.getVal(0) << "." << endl;
//      fail++;
//    }
//  normRand *= normRand;
//  CMatrix var = meanCol(meanRow(normRand));
//  if(abs(var.getVal(0)-1)<1e-2)
//    cout << "randn variance matches." << endl;
//  else
//    {
//      cout << "POSSIBLE FAILURE: randn variance " << var.getVal(0) << "." << endl;
//      fail++;
//    }

//  // TODO Check variance as well ... not sure of best way of checking randn!
//  return fail;
//}
//int testGemm()
//{
//  int fail = 0;
//  // gemm test
//  CMatrix D;
//  D.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "D");
//  CMatrix E;
//  E.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "E");
//  CMatrix F;
//  F.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "F");
//  CMatrix G;
//  G.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "G");
//  CMatrix H;
//  H.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "H");
//  CMatrix alph;
//  alph.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "alpha");
//  double alpha = alph.getVal(0);
//  CMatrix bet;
//  bet.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "beta");
//  double beta = bet.getVal(0);

//  // "n" "n" test
//  CMatrix GEMM1;
//  GEMM1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "GEMM1");
//  F.gemm(D, E, alpha, beta, "n", "N");
//  if(F.equals(GEMM1))
//    cout << "gemm nn matches." << endl;
//  else
//    {
//      cout << "FAILURE: gemm nn." << endl;
//      fail++;
//    }

//  // "t" "t" test
//  CMatrix GEMM2;
//  GEMM2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "GEMM2");
//  G.gemm(D, E, alpha, beta, "t", "T");
//  if(G.equals(GEMM2))
//    cout << "gemm tt matches." << endl;
//  else
//    {
//      cout << "FAILURE: gemm tt." << endl;
//      fail++;
//    }
//  CMatrix GEMM3;
//  GEMM3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "GEMM3");
//  GEMM1.gemm(D, H, alpha, beta, "N", "t");
//  if(GEMM1.equals(GEMM3))
//    cout << "gemm nt matches." << endl;
//  else
//    {
//      cout << "FAILURE: gemm tt." << endl;
//      fail++;
//    }
//  CMatrix GEMM4;
//  GEMM4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemmMatrixTest.mat", "GEMM4");
//  GEMM2.gemm(D, H, alpha, beta, "T", "n");
//  if(GEMM2.equals(GEMM4))
//    cout << "gemm tn matches." << endl;
//  else
//    {
//      cout << "FAILURE: gemm tt." << endl;
//      fail++;
//    }
//  return fail;
//}
//int testSyrk()
//{
//  int fail = 0;
//  // syrk test
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "A");
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "C");
//  C.setSymmetric(true);
//  CMatrix D;
//  D.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "D");
//  D.setSymmetric(true);
//  CMatrix alph;
//  alph.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "alpha");
//  double alpha = alph.getVal(0);
//  CMatrix bet;
//  bet.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "beta");
//  double beta = bet.getVal(0);
//  CMatrix SYRK1;
//  SYRK1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "SYRK1");
//  CMatrix SYRK2;
//  SYRK2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrkMatrixTest.mat", "SYRK2");
//  CMatrix F;

//  F.deepCopy(C);
//  F.syrk(A, alpha, beta, "u", "N");
//  if(F.equals(SYRK1))
//    cout << "syrk un matches." << endl;
//  else
//    {
//      cout << "FAILURE: syrk un." << endl;
//      fail++;
//    }
//  F.deepCopy(C);
//  F.syrk(A, alpha, beta, "L", "n");
//  if(F.equals(SYRK1))
//    cout << "syrk ln matches." << endl;
//  else
//    {
//      cout << "FAILURE: syrk ln." << endl;
//      fail++;
//    }
//  F.deepCopy(D);
//  F.syrk(A, alpha, beta, "U", "t");
//  if(F.equals(SYRK2))
//    cout << "syrk ut matches." << endl;
//  else
//    {
//      cout << "FAILURE: syrk ut." << endl;
//      fail++;
//    }
//  F.deepCopy(D);
//  F.syrk(A, alpha, beta, "l", "T");
//  if(F.equals(SYRK2))
//    cout << "syrk lt matches." << endl;
//  else
//    {
//      cout << "FAILURE: syrk lt." << endl;
//      fail++;
//    }
//  return fail;
//}
//int testTrmm()
//{
//  int fail = 0;
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "B");
//  CMatrix alph;
//  alph.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "alpha");
//  double alpha = alph.getVal(0);
//  CMatrix L;
//  L.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "L");
//  L.setTriangular(true);
//  CMatrix L2;
//  L2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "L2");
//  L2.setTriangular(true);
//  CMatrix U;
//  U.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "U");
//  U.setTriangular(true);
//  CMatrix U2;
//  U2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "U2");
//  U2.setTriangular(true);
//  CMatrix F;

//  CMatrix TRMM1;
//  TRMM1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM1");
//  F.deepCopy(B);
//  F.trmm(L, alpha, "L", "L", "N", "N");
//  if(F.equals(TRMM1))
//    cout << "trmm llnn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm llnn." << endl;
//      fail++;
//    }

//  CMatrix TRMM2;
//  TRMM2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM2");
//  F.deepCopy(B);
//  F.trmm(L, alpha, "L", "L", "T", "N");
//  if(F.equals(TRMM2))
//    cout << "trmm lltn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm lltn." << endl;
//      fail++;
//    }

//  CMatrix TRMM3;
//  TRMM3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM3");
//  F.deepCopy(B);
//  F.trmm(L2, alpha, "R", "L", "N", "N");
//  if(F.equals(TRMM3))
//    cout << "trmm rlnn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm rlnn." << endl;
//      fail++;
//    }
//  CMatrix TRMM4;
//  TRMM4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM4");
//  F.deepCopy(B);
//  F.trmm(L2, alpha, "R", "L", "T", "N");
//  if(F.equals(TRMM4))
//    cout << "trmm rltn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm rltn." << endl;
//      fail++;
//    }

//  CMatrix TRMM5;
//  TRMM5.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM5");
//  F.deepCopy(B);
//  F.trmm(L, alpha, "L", "L", "N", "U");
//  if(F.equals(TRMM5))
//    cout << "trmm llnu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm llnu." << endl;
//      fail++;
//    }

//  CMatrix TRMM6;
//  TRMM6.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM6");
//  F.deepCopy(B);
//  F.trmm(L, alpha, "L", "L", "T", "U");
//  if(F.equals(TRMM6))
//    cout << "trmm lltu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm lltu." << endl;
//      fail++;
//    }

//  CMatrix TRMM7;
//  TRMM7.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM7");
//  F.deepCopy(B);
//  F.trmm(L2, alpha, "R", "L", "N", "U");
//  if(F.equals(TRMM7))
//    cout << "trmm rlnu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm rlnu." << endl;
//      fail++;
//    }
//  CMatrix TRMM8;
//  TRMM8.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM8");
//  F.deepCopy(B);
//  F.trmm(L2, alpha, "R", "L", "T", "U");
//  if(F.equals(TRMM8))
//    cout << "trmm rltu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm rltu." << endl;
//      fail++;
//    }
//  CMatrix TRMM9;
//  TRMM9.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM9");
//  F.deepCopy(B);
//  F.trmm(U, alpha, "L", "U", "N", "N");
//  if(F.equals(TRMM9))
//    cout << "trmm lunn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm lunn." << endl;
//      fail++;
//    }

//  CMatrix TRMM10;
//  TRMM10.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM10");
//  F.deepCopy(B);
//  F.trmm(U, alpha, "L", "U", "T", "N");
//  if(F.equals(TRMM10))
//    cout << "trmm lutn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm lutn." << endl;
//      fail++;
//    }

//  CMatrix TRMM11;
//  TRMM11.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM11");
//  F.deepCopy(B);
//  F.trmm(U2, alpha, "R", "U", "N", "N");
//  if(F.equals(TRMM11))
//    cout << "trmm runn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm runn." << endl;
//      fail++;
//    }
//  CMatrix TRMM12;
//  TRMM12.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM12");
//  F.deepCopy(B);
//  F.trmm(U2, alpha, "R", "U", "T", "N");
//  if(F.equals(TRMM12))
//    cout << "trmm rutn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm rutn." << endl;
//      fail++;
//    }

//  CMatrix TRMM13;
//  TRMM13.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM13");
//  F.deepCopy(B);
//  F.trmm(U, alpha, "L", "U", "N", "U");
//  if(F.equals(TRMM13))
//    cout << "trmm lunu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm lunu." << endl;
//      fail++;
//    }

//  CMatrix TRMM14;
//  TRMM14.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM14");
//  F.deepCopy(B);
//  F.trmm(U, alpha, "L", "U", "T", "U");
//  if(F.equals(TRMM14))
//    cout << "trmm lutu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm lutu." << endl;
//      fail++;
//    }

//  CMatrix TRMM15;
//  TRMM15.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM15");
//  F.deepCopy(B);
//  F.trmm(U2, alpha, "R", "U", "N", "U");
//  if(F.equals(TRMM15))
//    cout << "trmm runu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm runu." << endl;
//      fail++;
//    }
//  CMatrix TRMM16;
//  TRMM16.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trmmMatrixTest.mat", "TRMM16");
//  F.deepCopy(B);
//  F.trmm(U2, alpha, "R", "U", "T", "U");
//  if(F.equals(TRMM16))
//    cout << "trmm rutu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trmm rutu." << endl;
//      fail++;
//    }


//  return fail;
//}
//int testTrsm()
//{
//  double tolInv = 1e-8;
//  int fail = 0;
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "B");
//  CMatrix alph;
//  alph.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "alpha");
//  double alpha = alph.getVal(0);
//  CMatrix L;
//  L.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "L");
//  L.setTriangular(true);
//  CMatrix L2;
//  L2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "L2");
//  L2.setTriangular(true);
//  CMatrix U;
//  U.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "U");
//  U.setTriangular(true);
//  CMatrix U2;
//  U2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "U2");
//  U2.setTriangular(true);
//  CMatrix F;

//  CMatrix TRSM1;
//  TRSM1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM1");
//  F.deepCopy(B);
//  F.trsm(L, alpha, "L", "L", "N", "N");
//  double absDiff=F.maxAbsDiff(TRSM1);
//  if(absDiff<tolInv)
//    cout << "trsm llnn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm llnn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM2;
//  TRSM2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM2");
//  F.deepCopy(B);
//  F.trsm(L, alpha, "L", "L", "T", "N");
//  absDiff=F.maxAbsDiff(TRSM2);
//  if(absDiff<tolInv)
//    cout << "trsm lltn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm lltn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM3;
//  TRSM3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM3");
//  F.deepCopy(B);
//  F.trsm(L2, alpha, "R", "L", "N", "N");
//  absDiff=F.maxAbsDiff(TRSM3);
//  if(absDiff<tolInv)
//    cout << "trsm rlnn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm rlnn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }
//  CMatrix TRSM4;
//  TRSM4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM4");
//  F.deepCopy(B);
//  F.trsm(L2, alpha, "R", "L", "T", "N");
//  absDiff=F.maxAbsDiff(TRSM4);
//  if(absDiff<tolInv)
//    cout << "trsm rltn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm rltn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM5;
//  TRSM5.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM5");
//  F.deepCopy(B);
//  F.trsm(L, alpha, "L", "L", "N", "U");
//  absDiff=F.maxAbsDiff(TRSM5);
//  if(absDiff<tolInv)
//    cout << "trsm llnu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm llnu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM6;
//  TRSM6.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM6");
//  F.deepCopy(B);
//  F.trsm(L, alpha, "L", "L", "T", "U");
//  absDiff=F.maxAbsDiff(TRSM6);
//  if(absDiff<tolInv)
//    cout << "trsm lltu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm lltu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM7;
//  TRSM7.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM7");
//  F.deepCopy(B);
//  F.trsm(L2, alpha, "R", "L", "N", "U");
//  absDiff=F.maxAbsDiff(TRSM7);
//  if(absDiff<tolInv)
//    cout << "trsm rlnu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm rlnu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }
//  CMatrix TRSM8;
//  TRSM8.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM8");
//  F.deepCopy(B);
//  F.trsm(L2, alpha, "R", "L", "T", "U");
//  absDiff=F.maxAbsDiff(TRSM8);
//  if(absDiff<tolInv)
//    cout << "trsm rltu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm rltu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }
//  CMatrix TRSM9;
//  TRSM9.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM9");
//  F.deepCopy(B);
//  F.trsm(U, alpha, "L", "U", "N", "N");
//  absDiff=F.maxAbsDiff(TRSM9);
//  if(absDiff<tolInv)
//    cout << "trsm lunn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm lunn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM10;
//  TRSM10.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM10");
//  F.deepCopy(B);
//  F.trsm(U, alpha, "L", "U", "T", "N");
//  absDiff=F.maxAbsDiff(TRSM10);
//  if(absDiff<tolInv)
//    cout << "trsm lutn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm lutn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM11;
//  TRSM11.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM11");
//  F.deepCopy(B);
//  F.trsm(U2, alpha, "R", "U", "N", "N");
//  absDiff=F.maxAbsDiff(TRSM11);
//  if(absDiff<tolInv)
//    cout << "trsm runn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm runn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }
//  CMatrix TRSM12;
//  TRSM12.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM12");
//  F.deepCopy(B);
//  F.trsm(U2, alpha, "R", "U", "T", "N");
//  absDiff=F.maxAbsDiff(TRSM12);
//  if(absDiff<tolInv)
//    cout << "trsm rutn matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm rutn, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM13;
//  TRSM13.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM13");
//  F.deepCopy(B);
//  F.trsm(U, alpha, "L", "U", "N", "U");
//  absDiff=F.maxAbsDiff(TRSM13);
//  if(absDiff<tolInv)
//    cout << "trsm lunu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm lunu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM14;
//  TRSM14.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM14");
//  F.deepCopy(B);
//  F.trsm(U, alpha, "L", "U", "T", "U");
//  absDiff=F.maxAbsDiff(TRSM14);
//  if(absDiff<tolInv)
//    cout << "trsm lutu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm lutu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }

//  CMatrix TRSM15;
//  TRSM15.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM15");
//  F.deepCopy(B);
//  F.trsm(U2, alpha, "R", "U", "N", "U");
//  absDiff = F.maxAbsDiff(TRSM15);
//  if(absDiff<tolInv)
//    cout << "trsm runu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm runu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }
//  CMatrix TRSM16;
//  TRSM16.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "trsmMatrixTest.mat", "TRSM16");
//  F.deepCopy(B);
//  F.trsm(U2, alpha, "R", "U", "T", "U");
//  absDiff=F.maxAbsDiff(TRSM16);
//  if(absDiff<tolInv)
//    cout << "trsm rutu matches." << endl;
//  else
//    {
//      cout << "FAILURE: trsm rutu, absolute difference " << absDiff << "." << endl;
//      fail++;
//    }


//  return fail;
//}


//int testAxpy()
//{
//  int fail = 0;
//  CMatrix iMat;
//  iMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "i");
//  int i = (int)iMat.getVal(0) - 1;
//  CMatrix jMat;
//  jMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "j");
//  int j = (int)jMat.getVal(0) - 1;
//  CMatrix kMat;
//  kMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "k");
//  int k = (int)kMat.getVal(0) - 1;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "A");
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "B");
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "C");
//  CMatrix D;
//  D.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "D");
//  CMatrix AXPY1;
//  AXPY1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "AXPY1");
//  CMatrix F;
//  F.deepCopy(B);
//  F.axpyRowRow(i, A, k, alpha);
//  if(F.equals(AXPY1))
//    cout << "axpyRowRow matches." << endl;
//  else
//    {
//      cout << "FAILURE: axpyRowRow." << endl;
//      fail++;
//    }
//  CMatrix AXPY2;
//  AXPY2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "AXPY2");
//  F.deepCopy(B);
//  F.axpyRowCol(i, C, j, alpha);
//  if(F.equals(AXPY2))
//    cout << "axpyRowCol matches." << endl;
//  else
//    {
//      cout << "FAILURE: axpyRowCol." << endl;
//      fail++;
//    }
//  CMatrix AXPY3;
//  AXPY3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "AXPY3");
//  F.deepCopy(B);
//  F.axpyColCol(j, A, k, alpha);
//  if(F.equals(AXPY3))
//    cout << "axpyColCol matches." << endl;
//  else
//    {
//      cout << "FAILURE: axpyColCol." << endl;
//      fail++;
//    }
//  CMatrix AXPY4;
//  AXPY4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "AXPY4");
//  F.deepCopy(B);
//  F.axpyColRow(j, C, j, alpha);
//  if(F.equals(AXPY4))
//    cout << "axpyColRow matches." << endl;
//  else
//    {
//      cout << "FAILURE: axpyColRow." << endl;
//      fail++;
//    }

//  CMatrix AXPY5;
//  AXPY5.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "AXPY5");
//  F.deepCopy(D);
//  F.axpyDiagRow(C, j, alpha);
//  if(F.equals(AXPY5))
//    cout << "axpyDiagRow matches." << endl;
//  else
//    {
//      cout << "FAILURE: axpyDiagRow." << endl;
//      fail++;
//    }
//  CMatrix AXPY6;
//  AXPY6.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "axpyMatrixTest.mat", "AXPY6");
//  F.deepCopy(D);
//  F.axpyDiagCol(B, j, alpha);
//  if(F.equals(AXPY6))
//    cout << "axpyDiagCol matches." << endl;
//  else
//    {
//      cout << "FAILURE: axpyDiagCol." << endl;
//      fail++;
//    }
//  return fail;
//}

//int testGemv()
//{
//  int fail = 0;
//  CMatrix iMat;
//  iMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "i");
//  int i = (int)iMat.getVal(0) - 1;
//  CMatrix jMat;
//  jMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "j");
//  int j = (int)jMat.getVal(0) - 1;
//  CMatrix kMat;
//  kMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "k");
//  int k = (int)kMat.getVal(0) - 1;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix betaMat;
//  betaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "beta");
//  double beta = betaMat.getVal(0);
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "A");
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "B");
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "C");
//  CMatrix D;
//  D.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "D");
//  CMatrix E;
//  E.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "E");
//  CMatrix GEMV1;
//  GEMV1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV1");
//  CMatrix GEMV2;
//  GEMV2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV2");
//  CMatrix GEMV3;
//  GEMV3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV3");
//  CMatrix GEMV4;
//  GEMV4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV4");
//  CMatrix GEMV5;
//  GEMV5.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV5");
//  CMatrix GEMV6;
//  GEMV6.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV6");
//  CMatrix GEMV7;
//  GEMV7.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV7");
//  CMatrix GEMV8;
//  GEMV8.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gemvMatrixTest.mat", "GEMV8");

//  CMatrix F(B);
//  F.gemvRowRow(i, C, D, k, alpha, beta, "n");
//  if(F.equals(GEMV1))
//    cout << "gemvRowRow n matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvRowRow n." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvRowRow(i, E, D, k, alpha, beta, "t");
//  if(F.equals(GEMV2))
//    cout << "gemvRowRow t matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvRowRow t." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvRowCol(i, C, A, k, alpha, beta, "n");
//  if(F.equals(GEMV3))
//    cout << "gemvRowCol n matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvRowCol n." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvRowCol(i, E, A, k, alpha, beta, "t");
//  if(F.equals(GEMV4))
//    cout << "gemvRowCol t matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvRowCol t." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvColCol(j, C, A, k, alpha, beta, "n");
//  if(F.equals(GEMV5))
//    cout << "gemvColCol n matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvColCol n." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvColCol(j, E, A, k, alpha, beta, "t");
//  if(F.equals(GEMV6))
//    cout << "gemvColCol t matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvColCol t." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvColRow(j, C, D, k, alpha, beta, "n");
//  if(F.equals(GEMV7))
//    cout << "gemvColRow n matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvColRow n." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.gemvColRow(j, E, D, k, alpha, beta, "t");
//  if(F.equals(GEMV8))
//    cout << "gemvColRow t matches." << endl;
//  else
//  {
//      cout << "FAILURE: gemvColRow t." << endl;
//      fail++;
//  }
//  return fail;
//}
//int testGer()
//{
//  int fail = 0;
//  CMatrix iMat;
//  iMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "i");
//  int i = (int)iMat.getVal(0) - 1;
//  CMatrix jMat;
//  jMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "j");
//  int j = (int)jMat.getVal(0) - 1;
//  CMatrix kMat;
//  kMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "k");
//  int k = (int)kMat.getVal(0) - 1;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix x;
//  x.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "x");
//  CMatrix y;
//  y.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "y");
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "A");
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "B");
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "C");
//  CMatrix D;
//  D.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "D");
//  CMatrix GER1;
//  GER1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "GER1");
//  CMatrix GER2;
//  GER2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "GER2");
//  CMatrix GER3;
//  GER3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "GER3");
//  CMatrix GER4;
//  GER4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "GER4");
//  CMatrix GER5;
//  GER5.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "gerMatrixTest.mat", "GER5");

//  CMatrix F;
//  F.deepCopy(A);
//  F.ger(x, y, alpha);
//  if(F.equals(GER1))
//    cout << "ger matches." << endl;
//  else
//  {
//      cout << "FAILURE: ger." << endl;
//      fail++;
//  }
//  F.deepCopy(A);
//  F.gerRowRow(C, i, B, k, alpha);
//  if(F.equals(GER2))
//    cout << "gerRowRow matches." << endl;
//  else
//  {
//      cout << "FAILURE: gerRowRow." << endl;
//      fail++;
//  }
//  F.deepCopy(A);
//  F.gerRowCol(C, i, D, j, alpha);
//  if(F.equals(GER3))
//    cout << "gerRowCol matches." << endl;
//  else
//    {
//      cout << "FAILURE: gerRowCol." << endl;
//      fail++;
//    }
//  F.deepCopy(A);
//  F.gerColCol(B, j, D, k, alpha);
//  if(F.equals(GER4))
//    cout << "gerColCol matches." << endl;
//  else
//    {
//      cout << "FAILURE: gerColCol." << endl;
//      fail++;
//    }
//  F.deepCopy(A);
//  F.gerColRow(B, j, B, k, alpha);
//  if(F.equals(GER5))
//    cout << "gerColRow matches." << endl;
//  else
//    {
//      cout << "FAILURE: gerColRow." << endl;
//      fail++;
//    }
//  return fail;

//}

//int testSyr()
//{
//  int fail = 0;
//  CMatrix iMat;
//  iMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "i");
//  int i = (int)iMat.getVal(0) - 1;
//  CMatrix jMat;
//  jMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "j");
//  int j = (int)jMat.getVal(0) - 1;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix x;
//  x.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "x");
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "A");
//  A.setSymmetric(true);
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "B");
//  CMatrix SYR1;
//  SYR1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "SYR1");
//  CMatrix SYR2;
//  SYR2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "SYR2");
//  CMatrix SYR3;
//  SYR3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syrMatrixTest.mat", "SYR3");

//  CMatrix F;
//  F.deepCopy(A);
//  F.syr(x, alpha, "U");
//  if(F.equals(SYR1))
//    cout << "syr u matches." << endl;
//  else
//  {
//      cout << "FAILURE: syr u." << endl;
//      fail++;
//  }
//  F.deepCopy(A);
//  F.syrRow(B, i, alpha, "U");
//  if(F.equals(SYR2))
//    cout << "syrRow u matches." << endl;
//  else
//  {
//      cout << "FAILURE: syrRow u." << endl;
//      fail++;
//  }
//  F.deepCopy(A);
//  F.syrRow(B, i, alpha, "L");
//  if(F.equals(SYR2))
//    cout << "syrRow l matches." << endl;
//  else
//  {
//      cout << "FAILURE: syrRow l." << endl;
//      fail++;
//  }
//  F.deepCopy(A);
//  F.syrCol(B, j, alpha, "u");
//  if(F.equals(SYR3))
//    cout << "syrCol u matches." << endl;
//  else
//    {
//      cout << "FAILURE: syrCol u." << endl;
//      fail++;
//    }
//  F.deepCopy(A);
//  F.syrCol(B, j, alpha, "l");
//  if(F.equals(SYR3))
//    cout << "syrCol l matches." << endl;
//  else
//    {
//      cout << "FAILURE: syrCol l." << endl;
//      fail++;
//    }
//  return fail;

//}
//int testSymm()
//{
//  int fail = 0;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix betaMat;
//  betaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "beta");
//  double beta = betaMat.getVal(0);
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "A");
//  A.setSymmetric(true);
//  CMatrix B1;
//  B1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "B1");
//  CMatrix B2;
//  B2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "B2");
//  CMatrix C1;
//  C1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "C1");
//  CMatrix C2;
//  C2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "C2");
//  CMatrix SYMM1;
//  SYMM1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "SYMM1");
//  CMatrix SYMM2;
//  SYMM2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "SYMM2");
//  CMatrix SYMM3;
//  SYMM3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "SYMM3");
//  CMatrix SYMM4;
//  SYMM4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symmMatrixTest.mat", "SYMM4");
//  CMatrix F(C1);
//  F.symm(A, B1, alpha, beta, "u", "l");
//  if(F.equals(SYMM1))
//    cout << "symm u l matches." << endl;
//  else
//  {
//      cout << "FAILURE: symm u l." << endl;
//      fail++;
//  }
//  F.deepCopy(C2);
//  F.symm(A, B2, alpha, beta, "U", "r");
//  if(F.equals(SYMM2))
//    cout << "symm u r matches." << endl;
//  else
//  {
//      cout << "FAILURE: symm u r." << endl;
//      fail++;
//  }
//  F.deepCopy(C1);
//  F.symm(A, B1, alpha, beta, "l", "L");
//  if(F.equals(SYMM3))
//    cout << "symm l l matches." << endl;
//  else
//  {
//      cout << "FAILURE: symm l l." << endl;
//      fail++;
//  }
//  F.deepCopy(C2);
//  F.symm(A, B2, alpha, beta, "L", "R");
//  if(F.equals(SYMM4))
//    cout << "symm l r matches." << endl;
//  else
//  {
//      cout << "FAILURE: symm l r." << endl;
//      fail++;
//  }
//  return fail;
//}
//int testSymv()
//{
//  int fail = 0;
//  CMatrix iMat;
//  iMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "i");
//  int i = (int)iMat.getVal(0) - 1;
//  CMatrix jMat;
//  jMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "j");
//  int j = (int)jMat.getVal(0) - 1;
//  CMatrix kMat;
//  kMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "k");
//  int k = (int)kMat.getVal(0) - 1;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix betaMat;
//  betaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "beta");
//  double beta = betaMat.getVal(0);
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "A");
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "B");
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "C");
//  C.setSymmetric(true);
//  CMatrix D;
//  D.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "D");
//  CMatrix SYMV1;
//  SYMV1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV1");
//  CMatrix SYMV2;
//  SYMV2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV2");
//  CMatrix SYMV3;
//  SYMV3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV3");
//  CMatrix SYMV4;
//  SYMV4.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV4");
//  CMatrix SYMV5;
//  SYMV5.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV5");
//  CMatrix SYMV6;
//  SYMV6.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV6");
//  CMatrix SYMV7;
//  SYMV7.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV7");
//  CMatrix SYMV8;
//  SYMV8.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "symvMatrixTest.mat", "SYMV8");

//  CMatrix F(B);
//  F.symvRowRow(i, C, D, k, alpha, beta, "u");
//  if(F.equals(SYMV1))
//    cout << "symvRowRow u matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvRowRow u." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvRowRow(i, C, D, k, alpha, beta, "l");
//  if(F.equals(SYMV2))
//    cout << "symvRowRow l matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvRowRow l." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvRowCol(i, C, A, k, alpha, beta, "u");
//  if(F.equals(SYMV3))
//    cout << "symvRowCol u matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvRowCol u." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvRowCol(i, C, A, k, alpha, beta, "l");
//  if(F.equals(SYMV4))
//    cout << "symvRowCol l matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvRowCol l." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvColCol(j, C, A, k, alpha, beta, "u");
//  if(F.equals(SYMV5))
//    cout << "symvColCol u matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvColCol u." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvColCol(j, C, A, k, alpha, beta, "l");
//  if(F.equals(SYMV6))
//    cout << "symvColCol l matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvColCol l." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvColRow(j, C, D, k, alpha, beta, "u");
//  if(F.equals(SYMV7))
//    cout << "symvColRow u matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvColRow u." << endl;
//      fail++;
//  }
//  F.deepCopy(B);
//  F.symvColRow(j, C, D, k, alpha, beta, "l");
//  if(F.equals(SYMV8))
//    cout << "symvColRow l matches." << endl;
//  else
//  {
//      cout << "FAILURE: symvColRow l." << endl;
//      fail++;
//  }
//  return fail;
//}
//int testSysv()
//{
//  int fail = 0;
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "sysvMatrixTest.mat", "A");
//  A.setSymmetric(true);
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "sysvMatrixTest.mat", "B");
//  CMatrix SYSV1;
//  SYSV1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "sysvMatrixTest.mat", "SYSV1");
//  CMatrix F(B);
//  CMatrix G(A);
//  F.sysv(G, "u");
//  if(F.equals(SYSV1))
//    cout << "sysv u matches." << endl;
//  else
//    {
//      cout << "FAILURE: sysv u." << endl;
//      fail++;
//    }
//  F.deepCopy(B);
//  G.deepCopy(A);
//  F.sysv(G, "l");
//  if(F.equals(SYSV1))
//    cout << "sysv l matches." << endl;
//  else
//    {
//      cout << "FAILURE: sysv l." << endl;
//      fail++;
//    }
//  return fail;
//}

//int testSyev()
//{
//  int fail = 0;
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syevMatrixTest.mat", "C");
//  C.setSymmetric(true);
//  CMatrix SYEV1;
//  SYEV1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syevMatrixTest.mat", "SYEV1");
//  CMatrix SYEV2;
//  SYEV2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syevMatrixTest.mat", "SYEV2");
//  CMatrix SYEV3;
//  SYEV3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "syevMatrixTest.mat", "SYEV3");

//  CMatrix F(C);
//  CMatrix D(F.getCols(), 1);
//  F.syev(D, "v", "u");
//  if(F.equals(SYEV2) && D.equals(SYEV3))
//    cout << "syev eigenvectors and eigenvalues match." << endl;
//  else
//  {
//    cout << "FAILURE: syev eigenvectors and eigenvalues." << endl;
//    fail++;
//  }
//  F.deepCopy(C);
//  D.zeros();
//  F.syev(D, "n", "u");
//  if(D.equals(SYEV3))
//    cout << "syev eigenvalues match." << endl;
//  else
//  {
//      cout << "FAILURE: syev eigenvalues." << endl;
//      fail++;
//  }
//  return fail;
//}

//int testScale()
//{
//  int fail = 0;
//  CMatrix iMat;
//  iMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "i");
//  int i = (int)iMat.getVal(0) - 1;
//  CMatrix jMat;
//  jMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "j");
//  int j = (int)jMat.getVal(0) - 1;
//  CMatrix alphaMat;
//  alphaMat.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "alpha");
//  double alpha = alphaMat.getVal(0);
//  CMatrix A;
//  A.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "A");
//  CMatrix B;
//  B.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "B");
//  CMatrix C;
//  C.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "C");
//  CMatrix SCALE1;
//  SCALE1.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "SCALE1");
//  CMatrix SCALE2;
//  SCALE2.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "SCALE2");
//  CMatrix SCALE3;
//  SCALE3.readMatlabFile("matfiles" + ndlstrutil::dirSep() + "scaleMatrixTest.mat", "SCALE3");

//  A.scale(alpha);
//  if(A.equals(SCALE1))
//    cout << "scale matches." << endl;
//  else
//  {
//      cout << "FAILURE: scale." << endl;
//      fail++;
//  }
//  B.scaleRow(i, alpha);
//  if(B.equals(SCALE2))
//    cout << "scaleRow matches." << endl;
//  else
//  {
//      cout << "FAILURE: scaleRow." << endl;
//      fail++;
//  }
//  C.scaleCol(j, alpha);
//  if(C.equals(SCALE3))
//    cout << "scaleCol matches." << endl;
//  else
//  {
//      cout << "FAILURE: scaleCol." << endl;
//      fail++;
//  }
//  return fail;


//}
///*
//  assert(max(A-Ainv)
//  cout << "A = " << endl << A;

//  // create a positive definite matrix
//  double b[9] = {18.0455,    4.9297,    3.6417,
//		 4.9297,    5.9624,   -4.7233,
//		 3.6417,   -4.7233,    7.8203};
//  CMatrix B(nrows, ncols, b);
//  cout << "Matrix B = " << endl << B;
//  CMatrix C(B);
////CMatrix C(nrows, ncols);
//  //  C.deepCopy(B);
//  //C.chol();
//  cout << "Cholesky decomposition of B = " << endl << C;
//  CMatrix D(nrows, ncols);
//  D.deepCopy(B);
//  D.pdinv();
//  cout << "Inverse of B using pdinv = " << endl << D;
//  CMatrix E(nrows, ncols);
//  E.deepCopy(B);
//  E.inv();
//  cout << "Inverse of B using inv = " << endl << E;
//  CMatrix F(nrows, ncols);
//  F.gemm(B, E, 1.0, 0.0, "n", "n");
//  cout << "B multiplied by its inverse = " << endl << F;
//  CMatrix G(nrows, ncols);
//  G.gemm(B, D, 1.0, 0.0, "n", "n");
//  cout << "B multiplied by its pdinverse = " << endl << G;
//  double trA = trace(A);
//  cout << "Trace of A is " << trA << endl;
//  cout << "B is " << B << endl;

//  double x[60] = {-0.4326,    0.2944,   -1.6041,
//		  -1.6656,   -1.3362,   0.2573,
//		  0.1253,    0.7143,   -1.0565,
//		  0.2877,    1.6236,    1.4151,
//		  -1.1465,   -0.6918,   -0.8051,
//		  1.1909,    0.8580,    0.5287,
//		  1.1892,    1.2540,    0.2193,
//		  -0.0376,   -1.5937,   -0.9219,
//		  0.3273,   -1.4410,   -2.1707,
//		  0.1746,    0.5711,   -0.0592,
//		  -0.1867,   -0.3999,   -1.0106,
//		  0.7258,    0.6900,    0.6145,
//		  -0.5883,    0.8156,    0.5077,
//		  2.1832,    0.7119,    1.6924,
//		  -0.1364,    1.2902,    0.5913,
//		  0.1139,    0.6686,   -0.6436,
//		  1.0668,    1.1908,    0.3803,
//		  0.0593,   -1.2025,   -1.0091,
//		  -0.0956,   -0.0198,   -0.0195,
//		  -0.8323,   -0.1567,   -0.0482};
//  nrows=3;
//  ncols=20;
//  CMatrix X(nrows, ncols, x);
//  X.trans();
//  ncols = 3;
//  nrows = 20;
//  cout << "X: " << endl << X << endl;
//  vector<int> rows;
//  rows.push_back(0);
//  rows.push_back(3);
//  rows.push_back(19);
//  cout << "Extracting  rows." << endl;
//  CMatrix X2(rows.size(), ncols);
//  X.getMatrix(X2, rows, 0, ncols-1);
//  cout << "Extracted matrix: " << endl << X2;
//  CMatrix X3(X.getRows(), X.getCols());
//  X3.deepCopy(X);
//  X3.appendCols(X);
//  X3.appendCols(X);
//  cout << X3.getCols() << endl;
//  cout << X3.getRows() << endl;
//  //  X3.appendRows(X);
//  cout << X3;
//  vector<int> cols;
//  cols.push_back(0);
//  cols.push_back(4);
//  cols.push_back(8);
//  CMatrix X4(nrows, cols.size());
//  X3.getMatrix(X4, 0, nrows-1, cols);
//  X4-=X;
//  cout<<X4;
//  CMatrix cov(ncols, ncols);
//  cov.gemm(X, X, 1.0, 0.0, "t", "n");
//  cout<< "Covariance " << endl << cov;

//  // now transpose x and use gemm with the same operation.
//  X.trans();
//  CMatrix cov2(ncols, ncols);
//  cov2.gemm(X, X, 1.0, 0.0, "n", "t");
//  cout<< "Covariance 2" << endl << cov2;

//  // this should be the same as the other 2 but it isn't"!
//  cov.dsyrk(X, "U", 1.0, 0.0, "n");
//  cout << "Covariance 3" << endl << cov;
//  cout << "Covariance 4" << endl << multiply(X, "n", X, "t") << endl;
//  CMatrix X5(X.getRows(), X.getCols());
//  X5.deepCopy(X);
//  X5.trans();
//  cout << "Covariance 5" << endl << multiply(X, X5) << endl;
//  cout << "Sum over all X: " << sum(X) << endl;

//  cout << "Kernel 1" << endl << multiply(X, "t", X, "n") << endl;

//  //  X.trans();
//  cout << "Row sum of X: " << endl <<sumRow(X) << endl;
//  cout << "Row mean of X: " << endl <<meanRow(X) << endl;
//  cout << "Col sum of X: " << endl << sumCol(X) << endl;
//  cout << "Col mean of X: " << endl << meanCol(X) << endl;

//  CMatrix XBar(X.getRows(), X.getCols());
//  XBar.deepCopy(X);
//  XBar-=meanRow(X);
//  cout << "Centred X: " << endl << XBar << endl;
//  cout << "Mean of XBar: " << endl << meanRow(XBar) << endl;

//  CMatrix randMat(100000, 10);
//  randMat.randn();
//  //  cout << "random matrix: " << endl << randMat << endl;
//  cout << "mean of matrix: " << endl << meanRow(randMat) << endl;
//  randMat *= randMat;
//  cout << "var of matrix: " << endl << meanRow(randMat) << endl;

//  // test transpose and swap column and row.
//  double x2[21] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17, 18, 19, 20};
//  nrows=3;
//  ncols=7;
//  CMatrix X6(nrows, ncols, x2);
//  CMatrix X7(nrows, ncols, x2);
//  X5.trans();
//  cout << X6;
//  cout << endl << " swap rows 1 and 3 " << endl;
//  X6.swapRows(0, 2);
//  cout << X6;
//  X6.trans();
//  cout <<endl << " Transpose of matrix " << endl;
//  cout << X6;
//  cout << endl << " swap columns 1 and 3 " << endl;
//  X6.swapCols(0, 2);
//  cout << X6;
//  X6.trans();
//  cout << endl << " Transpose of matrix " << endl;
//  cout << X6;
//  X6.subtract(X7);
//  assert(abs(max(X6))<EPS);

//  // Take x7, scale 3rd column by 2 transpose scale 3rd row by 0.5.
//  X7.trans();
//  CMatrix X8(X7);
//  X8.scale(2.0);
//  cout <<  X7.doCol(2, X8, 0);
//*/

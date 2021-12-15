/*This file contains interfaces to subroutines used by the bounded version of the LBFGS FORTRAN function

  21/10/05 bool* work changed to int* work in dxpose*/
#ifndef NDLFORTRAN_LBFGS_B_LINPACK_H
#define NDLFORTRAN_LBFGS_B_LINPACK_H

extern "C" void dpofa_(double* a, const int lda, const int n, const int info);
extern "C" void dtrsl_(double* t, const int ldt, const int n, double* b, int job, int info);

#endif

/*This file contains interfaces to the bounded version of the LBFGS FORTRAN function

  21/10/05 bool* work changed to int* work in dxpose*/
#ifndef NDLFORTRAN_LBFGS_B_H
#define NDLFORTRAN_LBFGS_B_H

#include "lapack.h"

extern "C" void timer_(double& ttime);
extern "C" void dpofa_(double* a, const int lda, const int n, const int info);
extern "C" void dtrsl_(double* t, const int ldt, const int n, double* b, int job, int info);


// From the website for the L-BFGS-B ('B' for Bounded) code (from http://www.ece.northwestern.edu/~nocedal/lbfgsb.html)
// Taken from Scipy version 1.7.1
// https://github.com/scipy/scipy/tree/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/optimize/lbfgsb_src
extern "C" void setulb_(
            const int& n,
            const int& m,  // number of corrections
			double* x, // length n
			const double* l, // length n
			const double* u, // length n
            const int* nbd, // length n, integer flags coding bounds (0: none, 1: lower, 2: both, 3: upper) for each variable
			const double& f, // function value, not clear whether this is modified within lbgsb
			const double* g, // gradient, not clear whether this is modified within lbgsb
			const double& factr, // solution accuracy based on function values
			const double& pgtol, // solution accuracy based on gradient values
			double* wa, // length (2mmax + 5)nmax + 12mmax^2 + 12mmax
            int* iwa, // length 3nmax
			char* task,  // length 60
            const int& iprint,  // verbosity
			char* csave, // message
			bool* lsave, // flags for controlling constraints and other algorithm settings
            int* isave, // integer array of length 44, contains diagnostic info
			double* dsave, // working array, length 29
            int* maxls  // max number of line search iterations
			);
            // integer intent(in),optional,check(len(x)>=n),depend(x) :: n=len(x)
            // integer intent(in) :: m
            // double precision dimension(n),intent(inout) :: x
            // double precision dimension(n),depend(n),intent(in) :: l
            // double precision dimension(n),depend(n),intent(in) :: u
            // integer dimension(n),depend(n),intent(in) :: nbd
            // double precision intent(inout) :: f
            // double precision dimension(n),depend(n),intent(inout) :: g
            // double precision intent(in) :: factr
            // double precision intent(in) :: pgtol
            // double precision dimension(2*m*n+5*n+11*m*m+8*m),depend(n,m),intent(inout) :: wa
            // integer dimension(3 * n),depend(n),intent(inout) :: iwa
            // character*60 intent(inout) :: task
            // integer intent(in) :: iprint
            // character*60 intent(inout) :: csave
            // logical dimension(4),intent(inout) :: lsave
            // integer dimension(44),intent(inout) :: isave
            // double precision dimension(29),intent(inout) :: dsave
            // integer intent(in) :: maxls	


// extern "C" void lbfgs_(const int& numVariables, 
// 		       const int& numCorrections,
// 		       double* X,
// 		       const double& funcVal,   // set by user to be func val.
// 		       const double* gradVals,  // set by user to be grad vals.
// 		       const int& diagCo,
// 		       const double* diag,
// 		       const int iPrint[2],
// 		       const double& prec,
// 		       const double& xtol,
// 		       double* W, // work vector size N(2M+1) + 2M
// 		       int& iFlag);

#endif

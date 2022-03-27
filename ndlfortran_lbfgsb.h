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
            C2F_CONST_INT n,
            C2F_CONST_INT m,  // number of corrections
			double* x, // length n
			const double* l, // length n
			const double* u, // length n
            const C2F_BASE_INT_TYPE* nbd, // length n, integer flags coding bounds (0: none, 1: lower, 2: both, 3: upper) for each variable
			const double& f, // function value, not clear whether this is modified within lbgsb
			const double* g, // gradient, not clear whether this is modified within lbgsb
			const double& factr, // solution accuracy based on function values
			const double& pgtol, // solution accuracy based on gradient values
			double* wa, // length (2mmax + 5)nmax + 12mmax^2 + 12mmax
            C2F_BASE_INT_TYPE* iwa, // length 3nmax
			char* task,  // length 60
            C2F_CONST_INT iprint,  // verbosity
			char* csave, // message
            #ifdef _WIN
            C2F_BASE_INT_TYPE* lsave, // boolean flags for controlling constraints and other algorithm settings
            #else
            bool* lsave, // boolean flags for controlling constraints and other algorithm settings
            #endif

            C2F_BASE_INT_TYPE* isave, // integer array of length 44, contains diagnostic info
			double* dsave, // working array, length 29
            C2F_INT maxls  // max number of line search iterations
			);

#endif


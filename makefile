#
#  Top Level Makefile for GPLVM
#  Version 0.11
#  July 6, 2005
#  Dec 23, 2008
# dependencies created with gcc -MM XXX.cpp

include make.linux

# how to print with makefile (write a variable):
# $(warning INCLUDE_DIR)target: $(INCLUDE_DIR)
# $(warning A top-level warning)

# FOO := $(warning ${INCLUDE_DIR})bar
# FOO2 := $(warning ${BOOSTLIB_INCLUDE})bar
#FOO := $(warning Right-hand side of a simple variable)bar
# BAZ = $(warning Right-hand side of a recursive variable)boo

# $(warning A target)target: $(warning $(INCLUDE_DIR))makefile $(INCLUDE_DIR)
# 		$(warning In a command script)
# 		ls

all: gplvm ivm gp libgp$(LIBSEXT) libgp$(LIBDEXT)

gplvm: gplvm.o CClctrl.o CGplvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gplvm gplvm.o CGplvm.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

gplvm.o: gplvm.cpp gplvm.h ndlexceptions.h ndlstrutil.h CMatrix.h \
  ndlassert.h CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h \
  CTransform.h CDataModel.h CDist.h CGplvm.h CMltools.h COptimisable.h \
  CNoise.h CClctrl.h
	$(CC) -c gplvm.cpp -o gplvm.o $(CCFLAGS)

ivm: ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o ivm  ivm.o CClctrl.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

ivm.o: ivm.cpp CIvm.h CKern.h CMatrix.h ivm.h CClctrl.h
	$(CC) -c ivm.cpp -o ivm.o $(CCFLAGS)

gp: gp.o CClctrl.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o gp gp.o CGp.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o $(LDFLAGS)

libgp$(LIBSEXT): CClctrl.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LIBSCOMMAND) libgp$(LIBSEXT) CGp.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o

libgp$(LIBDEXT): CClctrl.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o
	$(LIBDCOMMAND) libgp$(LIBDEXT) CGp.o CClctrl.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o ndlassert.o

gp.o: gp.cpp gp.h ndlexceptions.h ndlstrutil.h CMatrix.h ndlassert.h \
  CNdlInterfaces.h ndlutil.h ndlfortran.h lapack.h CKern.h CTransform.h \
  CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h CClctrl.h
	$(CC) -c gp.cpp -o gp.o $(CCFLAGS)


test_optim: test_optim.o
	$(LD) ${XLINKERFLAGS} -o test_optim test_optim.o $(LDFLAGS)

test_optim.o: test_optim.cpp
	$(CC) -c test_optim.cpp -o test_optim.o $(CCFLAGS)

Logger.o: Logger.cpp Logger.h
	$(CC) -c Logger.cpp -o Logger.o $(CCFLAGS)

bayesPlotUtil.o: bayesPlotUtil.cpp bayesPlotUtil.h
	$(CC) -c bayesPlotUtil.cpp -o bayesPlotUtil.o $(CCFLAGS)

BayesTestFunction.o: BayesTestFunction.cpp BayesTestFunction.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h GridSearch.h
	$(CC) -c BayesTestFunction.cpp -o BayesTestFunction.o $(CCFLAGS)

GridSearch.o: GridSearch.cpp GridSearch.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h CClctrl.h
	$(CC) -c GridSearch.cpp -o GridSearch.o $(CCFLAGS)

# To compile tests, the MATLAB interface must be enabled (i.e. define _NDLMATLAB)
tests: testDist testGp testIvm testKern testMatrix testMltools testNdlutil testNoise testTransform  

testDist: testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testDist testDist.o CMatrix.o ndlfortran.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o $(LDFLAGS) 

testDist.o: testDist.cpp CDist.h CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testDist.cpp -o testDist.o $(CCFLAGS)

testGp: testGp.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o
	$(LD) ${XLINKERFLAGS} -o testGp  testGp.o CGp.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o $(LDFLAGS)

testGp.o: testGp.cpp CGp.h CKern.h CMatrix.h CClctrl.h
	$(CC) -c testGp.cpp -o testGp.o $(CCFLAGS)

testGp_sklearn: testGp_sklearn.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o sklearn_util.o
	$(LD) ${XLINKERFLAGS} -o testGp_sklearn  testGp_sklearn.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o sklearn_util.o $(LDFLAGS)

testGp_sklearn.o: testGp_sklearn.cpp CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h sklearn_util.h
	$(CC) -c testGp_sklearn.cpp -o testGp_sklearn.o $(CCFLAGS)

testCSearchComparison.o: testCSearchComparison.cpp CSearchComparison.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h sklearn_util.h GridSearch.h
	$(CC) -c testCSearchComparison.cpp -o testCSearchComparison.o $(CCFLAGS)

testCSearchComparison: testCSearchComparison.o CSearchComparison.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o sklearn_util.o ndlassert.o GridSearch.o
	$(LD) ${XLINKERFLAGS} -o testCSearchComparison testCSearchComparison.o CSearchComparison.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o sklearn_util.o GridSearch.o $(LDFLAGS)

ifeq ($(PLAT), _WIN)
testCSearchComparison_full: testCSearchComparison_full.o CSearchComparison.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o BayesTestFunction.o GridSearch.o
	$(LD) ${XLINKERFLAGS} -o testCSearchComparison_full testCSearchComparison_full.o CSearchComparison.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o BayesTestFunction.o GridSearch.o $(LDFLAGS)

testCSearchComparison_full.o: testCSearchComparison_full.cpp CSearchComparison.h testBayesianSearch.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h Logger.h BayesTestFunction.h version.h GridSearch.h
	$(CC) -c testCSearchComparison_full.cpp -o testCSearchComparison_full.o $(CCFLAGS)

testBayesianSearch: testBayesianSearch.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o BayesTestFunction.o GridSearch.o
	$(LD) ${XLINKERFLAGS} -o testBayesianSearch testBayesianSearch.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o BayesTestFunction.o GridSearch.o $(LDFLAGS)

testBayesianSearch.o: testBayesianSearch.cpp testBayesianSearch.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h Logger.h BayesTestFunction.h GridSearch.h version.h
	$(CC) -c testBayesianSearch.cpp -o testBayesianSearch.o $(CCFLAGS)

else
testCSearchComparison_full: testCSearchComparison_full.o CSearchComparison.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o bayesPlotUtil.o BayesTestFunction.o  GridSearch.o
	$(LD) ${XLINKERFLAGS} -o testCSearchComparison_full testCSearchComparison_full.o CSearchComparison.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o bayesPlotUtil.o BayesTestFunction.o GridSearch.o $(LDFLAGS)

testCSearchComparison_full.o: testCSearchComparison_full.cpp CSearchComparison.h testBayesianSearch.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h Logger.h bayesPlotUtil.h BayesTestFunction.h GridSearch.h version.h
	$(CC) -c testCSearchComparison_full.cpp -o testCSearchComparison_full.o $(CCFLAGS)

testBayesianSearch: testBayesianSearch.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o bayesPlotUtil.o BayesTestFunction.o GridSearch.o
	$(LD) ${XLINKERFLAGS} -o testBayesianSearch testBayesianSearch.o CBayesianSearch.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o Logger.o bayesPlotUtil.o BayesTestFunction.o GridSearch.o $(LDFLAGS)

testBayesianSearch.o: testBayesianSearch.cpp testBayesianSearch.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h Logger.h bayesPlotUtil.h BayesTestFunction.h GridSearch.h version.h
	$(CC) -c testBayesianSearch.cpp -o testBayesianSearch.o $(CCFLAGS)
endif

testIvm: testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o
	$(LD) ${XLINKERFLAGS} -o testIvm  testIvm.o CIvm.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o $(LDFLAGS)

testIvm.o: testIvm.cpp CIvm.h CKern.h CMatrix.h CClctrl.h 
	$(CC) -c testIvm.cpp -o testIvm.o $(CCFLAGS)

testKern: testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testKern testKern.o CMatrix.o ndlfortran.o CKern.o CTransform.o COptimisable.o CDist.o ndlutil.o ndlstrutil.o CClctrl.o $(LDFLAGS) 

testKern.o: testKern.cpp CKern.h CDist.h CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testKern.cpp -o testKern.o $(CCFLAGS)

testKern_sklearn: testKern_sklearn.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o sklearn_util.o
	$(LD) ${XLINKERFLAGS} -o testKern_sklearn  testKern_sklearn.o CGp.o CMatrix.o ndlfortran.o ndlfortran_timer.o ndlfortran_linpack.o ndlfortran_lbfgsb.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CKern.o CDist.o CClctrl.o CMltools.o ndlassert.o sklearn_util.o $(LDFLAGS)

testKern_sklearn.o: testKern_sklearn.cpp CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h sklearn_util.h
	$(CC) -c testKern_sklearn.cpp -o testKern_sklearn.o $(CCFLAGS)

testMatrix: testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testMatrix testMatrix.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o $(LDFLAGS) 

testMatrix.o: testMatrix.cpp CMatrix.h CClctrl.h
	$(CC) -c testMatrix.cpp  -o testMatrix.o $(CCFLAGS)

testMatrix_fortran: testMatrix_fortran.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o ndlassert.o
	$(LD) ${XLINKERFLAGS} -o testMatrix_fortran testMatrix_fortran.o CMatrix.o ndlfortran.o ndlstrutil.o ndlutil.o CClctrl.o ndlassert.o $(LDFLAGS)

testMatrix_fortran.o: testMatrix_fortran.cpp CMatrix.h CClctrl.h
	$(CC) -c testMatrix_fortran.cpp  -o testMatrix_fortran.o $(CCFLAGS)

testMltools: testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testMltools  testMltools.o CMltools.o CMatrix.o ndlfortran.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CClctrl.o $(LDFLAGS)

testMltools.o: testMltools.cpp CMltools.h CKern.h CMatrix.h CClctrl.h 
	$(CC) -c testMltools.cpp -o testMltools.o $(CCFLAGS)

testNdlutil: testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testNdlutil testNdlutil.o ndlutil.o ndlstrutil.o CMatrix.o ndlfortran.o CClctrl.o $(LDFLAGS)

testNdlutil.o: testNdlutil.cpp ndlutil.h CClctrl.h
	$(CC) -c testNdlutil.cpp -o testNdlutil.o $(CCFLAGS)

testNoise: testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testNoise  testNoise.o CMatrix.o ndlfortran.o CNoise.o ndlutil.o ndlstrutil.o CTransform.o COptimisable.o CDist.o CClctrl.o $(LDFLAGS)

testNoise.o: testNoise.cpp CNoise.h CMatrix.h CClctrl.h
	$(CC) -c testNoise.cpp -o testNoise.o $(CCFLAGS)

testTransform: testTransform.o CMatrix.o ndlfortran.o  CTransform.o ndlutil.o ndlstrutil.o CClctrl.o
	$(LD) ${XLINKERFLAGS} -o testTransform testTransform.o CMatrix.o ndlfortran.o CTransform.o ndlutil.o ndlstrutil.o CClctrl.o $(LDFLAGS) 

testTransform.o: testTransform.cpp CTransform.h CMatrix.h CClctrl.h
	$(CC) -c testTransform.cpp -o testTransform.o $(CCFLAGS)

CClctrl.o: CClctrl.cpp CClctrl.h ndlstrutil.h ndlexceptions.h ndlutil.h \
  ndlassert.h ndlfortran.h CMatrix.h CNdlInterfaces.h lapack.h
	$(CC) -c CClctrl.cpp -o CClctrl.o $(CCFLAGS)

CGplvm.o: CGplvm.cpp CGplvm.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CGplvm.cpp -o CGplvm.o $(CCFLAGS)

CNoise.o: CNoise.cpp CNoise.h ndlexceptions.h ndlutil.h ndlassert.h \
  ndlfortran.h ndlstrutil.h CMatrix.h CNdlInterfaces.h lapack.h \
  CTransform.h COptimisable.h CDist.h CKern.h CDataModel.h
	$(CC) -c CNoise.cpp -o CNoise.o $(CCFLAGS)

CKern.o: CKern.cpp CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CDataModel.h CDist.h
	$(CC) -c CKern.cpp -o CKern.o $(CCFLAGS)

CTransform.o: CTransform.cpp CTransform.h CMatrix.h ndlassert.h \
  ndlexceptions.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h
	$(CC) -c CTransform.cpp -o CTransform.o $(CCFLAGS)

COptimisable.o: COptimisable.cpp COptimisable.h CMatrix.h ndlassert.h \
  ndlexceptions.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h \
  lapack.h ndlfortran_lbfgsb.h
	$(CC) -c COptimisable.cpp -o COptimisable.o $(CCFLAGS)

CDist.o: CDist.cpp CDist.h CMatrix.h ndlassert.h ndlexceptions.h \
  CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CTransform.h
	$(CC) -c CDist.cpp -o CDist.o $(CCFLAGS)

CMatrix.o: CMatrix.cpp CMatrix.h ndlassert.h ndlexceptions.h \
  CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h
	$(CC) -c CMatrix.cpp -o CMatrix.o $(CCFLAGS)

CGp.o: CGp.cpp CGp.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CGp.cpp -o CGp.o $(CCFLAGS)

CIvm.o: CIvm.cpp CIvm.h CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CIvm.cpp -o CIvm.o $(CCFLAGS)

# apparently there was never a make rule for CMltools.o in the original GPc
# it was being correctly generated automatically on WSL2 Ubuntu 20.04 until
# some internal setting changed. Then it alone of all the object files
# was being compiled with a conda version of GCC and failing
CMltools.o: CMltools.cpp CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h
	$(CC) -c CMltools.cpp -o CMltools.o $(CCFLAGS)

sklearn_util.o: sklearn_util.cpp sklearn_util.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h lapack.h \
  CDataModel.h CDist.h
	$(CC) -c sklearn_util.cpp -o sklearn_util.o $(CCFLAGS)

CBayesianSearch.o: CBayesianSearch.cpp CBayesianSearch.h CGp.cpp CGp.h \
  CMltools.h ndlassert.h ndlexceptions.h \
  ndlstrutil.h COptimisable.h CMatrix.h CNdlInterfaces.h ndlutil.h \
  ndlfortran.h lapack.h CKern.h CTransform.h CDataModel.h CDist.h \
  CNoise.h GridSearch.h
	$(CC) -c CBayesianSearch.cpp -o CBayesianSearch.o $(CCFLAGS) ${BOOSTLIB}

CSearchComparison.o: CSearchComparison.cpp CSearchComparison.h CBayesianSearch.h CKern.h ndlassert.h ndlexceptions.h CTransform.h \
  CMatrix.h CNdlInterfaces.h ndlstrutil.h ndlutil.h ndlfortran.h ndlfortran_lbfgsb.h \
  lapack.h CDataModel.h CDist.h CGp.h CMltools.h COptimisable.h CNoise.h \
  CClctrl.h GridSearch.h
	$(CC) -c CSearchComparison.cpp -o CSearchComparison.o $(CCFLAGS)

ndlutil.o: ndlutil.cpp ndlutil.h ndlassert.h ndlexceptions.h ndlfortran.h
	$(CC) -c ndlutil.cpp -o ndlutil.o $(CCFLAGS)

ndlstrutil.o: ndlstrutil.cpp ndlstrutil.h ndlexceptions.h
	$(CC) -c ndlstrutil.cpp -o ndlstrutil.o $(CCFLAGS)

ndlassert.o: ndlassert.cpp ndlassert.h ndlexceptions.h
	$(CC) -c ndlassert.cpp -o ndlassert.o $(CCFLAGS)

# Collected FORTRAN utilities.
ndlfortran.o: ndlfortran.f
	$(FC) -c ndlfortran.f -o ndlfortran.o $(FCFLAGS)

ifeq (${PLAT}, _WIN)
ndlfortran_timer.o: ndlfortran_timer.f
	$(FC) -c ndlfortran_timer.f -o ndlfortran_timer.o $(FCFLAGS) -fdefault-integer-8

ndlfortran_linpack.o: ndlfortran_linpack.f
	$(FC) -c ndlfortran_linpack.f -o ndlfortran_linpack.o $(FCFLAGS) -fdefault-integer-8

# ndlfortran_linpack.f ndlfortran_timer.f
ndlfortran_lbfgsb.o: ndlfortran_lbfgsb.f
	$(FC) -c ndlfortran_lbfgsb.f -o ndlfortran_lbfgsb.o $(FCFLAGS) -fdefault-integer-8
else
ndlfortran_timer.o: ndlfortran_timer.f
	$(FC) -c ndlfortran_timer.f -o ndlfortran_timer.o $(FCFLAGS)

ndlfortran_linpack.o: ndlfortran_linpack.f
	$(FC) -c ndlfortran_linpack.f -o ndlfortran_linpack.o $(FCFLAGS)

# ndlfortran_linpack.f ndlfortran_timer.f
ndlfortran_lbfgsb.o: ndlfortran_lbfgsb.f
	$(FC) -c ndlfortran_lbfgsb.f -o ndlfortran_lbfgsb.o $(FCFLAGS)
endif

# version info
GIT_BRANCH=$(shell git symbolic-ref HEAD)
GIT_COMMIT=$(shell git describe --always --dirty)
GIT_URL=$(shell git config --get remote.origin.url)
ifeq ($(PLAT), _WIN) # module
version.h: ../../.git/modules/include/BayesGPc/index
else
version.h: .git/index
endif
	echo "#define GIT_BRANCH \"$(GIT_BRANCH)\"" > $@
	echo "#define GIT_COMMIT \"$(GIT_COMMIT)\"" >> $@
	echo "#define GIT_URL \"$(GIT_URL)\"" >> $@
# echo "$(INCLUDE_DIR)"  # not working

# compile these fortran files after make clean since they aren't compiling with 
# other make commands as expected, might be that including a header file [header.h]
# only signals to link with [head.o] which I'm not currently including for these 
# fortran files?
clean:
	rm *.o
	make ndlfortran_timer.o
	make ndlfortran_linpack.o


#include <string>
#include "CKern.h"
#include "CMatrix.h"
#include "cnpy.h"
#include "CNdlInterfaces.h"
#include <stdexcept>


CMatrix* readNpzFile(const string fileName, const string variableName);
void getSklearnKernels(CCmpndKern *kern, cnpy::npz_t npz_dict, CMatrix *MX, bool structureOnly);
CKern* getSklearnKernel(CMatrix* X, cnpy::npz_t npz_dict, string kernel_key, string param_key, bool structureOnly);
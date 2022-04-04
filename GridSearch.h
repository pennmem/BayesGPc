#include "CMatrix.h"
#include <functional>

CMatrix gridSearch(std::function<double(const CMatrix&)> fcn,
                   std::vector<CMatrix> grid_vals);

CMatrix gridSearch1D(std::function<double(const CMatrix&)> fcn,
                     std::vector<CMatrix> grid_vals);

CMatrix gridSearch2D(std::function<double(const CMatrix&)> fcn,
                     std::vector<CMatrix> grid_vals);

CMatrix gridSearch3D(std::function<double(const CMatrix&)> fcn,
                     std::vector<CMatrix> grid_vals);

CMatrix gridSearch4D(std::function<double(const CMatrix&)> fcn,
                     std::vector<CMatrix> grid_vals);

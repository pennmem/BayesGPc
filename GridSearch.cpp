#include "GridSearch.h"

CMatrix gridSearch(std::function<double(const CMatrix&)> fcn,
                   vector<CMatrix> grid_vals) {
    assert(grid_vals.size() > 0);
    CMatrix x(1, grid_vals.size());
    switch (grid_vals.size()) {
        case 1:
            x = gridSearch1D(fcn, grid_vals);
            break;
        case 2:
            x = gridSearch2D(fcn, grid_vals);
            break;
        case 3:
            x = gridSearch3D(fcn, grid_vals);
            break;
        case 4:
            x = gridSearch4D(fcn, grid_vals);
            break;
        default:
            throw std::runtime_error("Grid search not implemented for dimensionality higher than 4D.");
    }
//    cout << "before return gridSearch: x " << x << endl;
    return x;
}

// global maximizer over grid
CMatrix gridSearch1D(std::function<double(const CMatrix&)> fcn,
                     vector<CMatrix> grid_vals) {
    CMatrix x_best(1, grid_vals.size());
    CMatrix x(1, grid_vals.size());
    double y_best = -std::numeric_limits<double>::infinity();
    double y;
    for (int i = 0; i < grid_vals[0].getRows(); i++) {
        x(0, 0) = grid_vals[0].getVal(i);
        y = fcn(x);
        if (y > y_best) {
            y_best = y;
            x_best = x;
        }
    }
    return x_best;
}

CMatrix gridSearch2D(std::function<double(const CMatrix&)> fcn,
                     vector<CMatrix> grid_vals) {
    CMatrix x_best(1, grid_vals.size());
    CMatrix x(1, grid_vals.size());
    double y_best = -std::numeric_limits<double>::infinity();
    double y;
    for (int i = 0; i < grid_vals[0].getRows(); i++) {
        x(0, 0) = grid_vals[0].getVal(i);
        for (int j = 0; j < grid_vals[1].getRows(); j++) {
            x(0, 1) = grid_vals[1].getVal(j);
            y = fcn(x);
            if (y > y_best) {
                y_best = y;
                x_best = x;
            }
        }
    }
    return x_best;
}


CMatrix gridSearch3D(std::function<double(const CMatrix&)> fcn,
                     vector<CMatrix> grid_vals) {
    CMatrix x_best(1, grid_vals.size());
    CMatrix x(1, grid_vals.size());
    double y_best = -std::numeric_limits<double>::infinity();
    double y;
    for (int i = 0; i < grid_vals[0].getRows(); i++) {
        x(0, 0) = grid_vals[0].getVal(i);
        for (int j = 0; j < grid_vals[1].getRows(); j++) {
            x(0, 1) = grid_vals[1].getVal(j);
            for (int k = 0; k < grid_vals[2].getRows(); k++) {
                x(0, 2) = grid_vals[2].getVal(k);
                y = fcn(x);
                if (y > y_best) {
                    y_best = y;
                    x_best = x;
                }
            }
        }
    }
    return x_best;
}


CMatrix gridSearch4D(std::function<double(const CMatrix&)> fcn,
                     vector<CMatrix> grid_vals) {
    CMatrix x_best(1, grid_vals.size());
    CMatrix x(1, grid_vals.size());
    double y_best = -std::numeric_limits<double>::infinity();
    double y;
    for (int i = 0; i < grid_vals[0].getRows(); i++) {
        x(0, 0) = grid_vals[0].getVal(i);
        for (int j = 0; j < grid_vals[1].getRows(); j++) {
            x(0, 1) = grid_vals[1].getVal(j);
            for (int k = 0; k < grid_vals[2].getRows(); k++) {
                x(0, 2) = grid_vals[2].getVal(k);
                for (int h = 0; h < grid_vals[3].getRows(); h++) {
                    x(0, 3) = grid_vals[3].getVal(h);
                    y = fcn(x);
                    if (y > y_best) {
                        y_best = y;
                        x_best = x;
                    }
                }
            }
        }
    }
    return x_best;
}


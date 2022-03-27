#include "GridSearch.h"

CMatrix gridSearch(std::function<double(const CMatrix&)> fcn,
                   vector<CMatrix> grid_vals) {
    assert(grid_vals.size() < 2 && grid_vals.size() > 0);
    CMatrix x(1, grid_vals.size());
    switch (grid_vals.size()) {
        case 1:
            x = gridSearch1D(fcn, grid_vals);
            break;
//        case 2:
//            GridSearch2D();
//            break;
    }
//    cout << "before return gridSearch: x " << x << endl;
    return x;
}

// global maximizer over grid
CMatrix gridSearch1D(std::function<double(const CMatrix&)> fcn,
                     vector<CMatrix> grid_vals) {
    CMatrix x_best(1, grid_vals.size());
    CMatrix x(1, grid_vals.size());
    double y_best = -std::numeric_limits<double>::infinity();;
    double y;
    for (int i = 0; i < grid_vals[0].getRows(); i++) {
        x(0, 0) = grid_vals[0].getVal(i);
        y = fcn(x);
        if (y > y_best) {
            y_best = y;
            x_best = x;
        }
//        cout << "y " << y << "   x " << x << endl;
//        cout << "y_best " << y_best << "   x_best " << x_best << endl;
    }
//    cout << "before return gridSearch1D: y_best " << y_best << "   x_best " << x_best << endl;
    return x_best;
}


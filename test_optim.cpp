// #define OPTIM_ENABLE_ARMA_WRAPPERS
#define OPTIM_ENABLE_EIGEN_WRAPPERS

// notes on eigen link time error
// test_optim.cpp builds and runs without optim::de(), i.e. eigen is being found
// using the header_only_version of the optim library works. 
// I originally installed optim with armadillo, but was not able to reinstall with eigen afterwards,
// should try again after cleanly uninstalling optim

// #include <iostream>
// #include <random>
// #include <eigen3/Eigen/Dense>

// template<typename eT, int iTr, int iTc>
// using EigenMat = Eigen::Matrix<eT,iTr,iTc>;

// namespace optim
// {
//     using Mat_t = Eigen::MatrixXd;
//     using Vec_t = Eigen::VectorXd;
//     using RowVec_t = Eigen::Matrix<double,1,Eigen::Dynamic>;
//     using VecInt_t = Eigen::VectorXi;
// }
#include "optim.hpp"

// Compilation and linking:
// g++ -Wall -std=c++11 -O3 -march=native -ffp-contract=fast -I/path/to/[armadillo or eigen] -I/path/to/optim/include optim_de_ex.cpp -o optim_de_ex.out -L/path/to/optim/lib -loptim
// ./optim_de_ex.out

// e.g.
// with eigen
// g++ -Wall -std=c++11 -O3 -march=native -ffp-contract=fast -I/usr/local/include/eigen3 -I/usr/local/include/optim test_optim.cpp -o test_optim.out -L/usr/local/include/optim -loptim
// with armadillo
// g++ -Wall -std=c++11 -O3 -march=native -ffp-contract=fast -I/usr/local/include/armadillo -I/usr/local/include/optim test_optim.cpp -o test_optim.out -L/usr/local/include/optim -loptim

//
// Ackley function

double ackley_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
// double ackley_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);
    const double pi = 3.14159;

    double obj_val = -20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2*pi*x) + std::cos(2*pi*y)) ) + 22.718282L;

    //

    return obj_val;
}

int main()
{
    // initial values:
    // arma::vec x = arma::ones(2,1) + 1.0; // (2,2)
    // Eigen::VectorXd x = Eigen::VectorXd(2);
    optim::Vec_t x = Eigen::VectorXd(2);
    std::cout << typeid(x).name() << std::endl;
    // x.setZero();

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    bool success = false;
    // success = true;
    success = optim::de(x, ackley_fn, nullptr);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    if (success) {
        std::cout << "de: Ackley test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
    }

    std::cout << "\nde: solution to Ackley test:\n" << x << std::endl;
    // arma::cout << "\nde: solution to Ackley test:\n" << x << arma::endl;

    return 0;
}

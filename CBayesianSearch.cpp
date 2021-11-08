#include "CBayesianSearch.h"


// TODO move out to separate header
double expected_improvement(const CMatrix& x, const CGp& model, double y_b, double exp_bias) {
    double EI;

    CMatrix mu_mat(1, 1);
    CMatrix std_mat(1, 1);
    double y_best = y_b - exp_bias;

    model.out(mu_mat, std_mat, x);
    double mu = mu_mat.getVal(0, 0);
    double std = std_mat.getVal(0, 0);

    if (std > 0) {
        boost::math::normal norm(0, 1);
        double Z = (mu - y_best)/std;
        EI = (mu - y_best) * cdf(norm, Z) + std * pdf(norm, Z);
    }
    else {
        EI = 0;
    }
    return -EI;
}

struct EIOptimStruct {
    CGp model;
    double y_b;
    double exp_bias; 
};

// TODO write wrapper for converting between CMatrix and major armadillo/eigen types

// wrapper for eigen vector input x
double expected_improvement_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data) {
    EIOptimStruct* optfn_data = reinterpret_cast<EIOptimStruct*>(opt_data);
    CMatrix x_cmat(1, (int)(x.rows()), x.data());
    return expected_improvement(x_cmat, optfn_data->model, optfn_data->y_b, optfn_data->exp_bias);
}


CMatrix* BayesianSearchModel::get_next_sample() {
    CMatrix* x = new CMatrix(1, x_dim);
    
    // initial random samples
    // TODO upgrade to Latin hypercube sampling
    if (num_samples < initial_samples) {
        boost::random::uniform_real_distribution<> dist(0.0, 1.0);
        boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > gen(rng, dist);
        for (unsigned int i = 0; i < x_dim; i++) {
            x->setVal(bounds->getVal(i, 0) + gen() * (bounds->getVal(i, 1) - bounds->getVal(i, 0)), i);
        }
    }
    else {
        // set up model
        // TODO move model set up to separate function, only set up function once after first sample
        CGaussianNoise noiseInit(y_samples);
        for(int i=0; i<noiseInit.getOutputDim(); i++) {noiseInit.setBiasVal(0.0, i);}
        noiseInit.setParam(1e-6, noiseInit.getOutputDim());
        
        int iters = 10;
        bool outputScaleLearnt = false;
        int approxType = 0;

        // update Gaussian process model
        // TODO valid? want to free old gp memory if it was previously allocated
        // if (gp != nullptr) {delete gp;}
        // want persistent access to model after fitting and getting next sample in order to fit new model
        gp = CGp(&kern, &noiseInit, x_samples, approxType, -1, 3);
        gp.setBetaVal(1);
        gp.setScale(1.0);
        gp.setBias(0.0);
        gp.updateM();

        gp.setVerbosity(2);
        gp.setDefaultOptimiser(CGp::BFGS);  //options: SCG, CG, GD
        gp.setObjectiveTol(1e-6);
        gp.setParamTol(1e-6);
        gp.setOutputScaleLearnt(outputScaleLearnt);
        gp.optimise(iters);

        // optimize acquisition function
        Eigen::VectorXd x_optim = Eigen::VectorXd(x_dim);
        Eigen::VectorXd lower_bounds = Eigen::VectorXd(x_dim);
        Eigen::VectorXd upper_bounds = Eigen::VectorXd(x_dim);
        // TODO choose better/random initial values? not strictly needed for global optimization
        for (int i = 0; i < x_dim; i++) {
            x_optim[i] = (bounds->getVal(i, 0) + bounds->getVal(i, 1))/2.0;
            lower_bounds[i] = bounds->getVal(i, 0);
            upper_bounds[i] = bounds->getVal(i, 1);
        }
        optim::algo_settings_t optim_settings;
        optim_settings.print_level = 2;
        optim_settings.vals_bound = true;
        optim_settings.lower_bounds = lower_bounds;
        optim_settings.upper_bounds = upper_bounds;
        bool success;

        if (acq_func_name.compare("expected_improvement") == 0) {
            EIOptimStruct opt_data = {gp, y_best, exploration_bias};
            cout << "entered optim::de." << endl;
            success = optim::de(x_optim, expected_improvement_optim, &opt_data, optim_settings);
            cout << "left optim::de.    success:" << success << endl;
        }
        else {
            throw std::invalid_argument("Acquisition function " + acq_func_name + " not implemented. Current options: 'expected_improvement'");
        }

        if (success) {
            for (int i = 0; i < x_dim; i++) {
                x->setVal(x_optim[i], i);
            }
        }
        else {
            cout << "Optimization of acquisition function failed." << endl;
            throw std::runtime_error("Optimization of acquisition function " + acq_func_name + " failed.");
        }
    }
    return x;
}

void BayesianSearchModel::add_sample(const CMatrix& x, const CMatrix& y) {
    if (num_samples == 0) {
        x_samples->copy(x);
        y_samples->copy(y);
    }
    else {
        x_samples->appendRows(x);
        y_samples->appendRows(y);
    }
    if (y.getVal(0, 0) > y_best) {
        y_best = y.getVal(0, 0);
    }
    num_samples++;
}

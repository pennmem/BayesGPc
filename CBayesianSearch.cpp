#include "CBayesianSearch.h"
#include "GridSearch.h"


double expected_improvement(const CMatrix& x, const CGp& model, double y_b, double exp_bias) {
    double EI;

    CMatrix mu_mat(1, 1);
    CMatrix std_mat(1, 1);
    double y_best = y_b + exp_bias;

    model.out(mu_mat, std_mat, x);
    // use predictive mean uncertainty rather than sample uncertainty 
    // (which includes observation noise/white noise)
    // model.out_sem(mu_mat, std_mat, x);
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
    return EI;
}

struct EIOptimStruct {
    CGp model;
    double y_b;
    double exp_bias;
};


#ifndef _WIN
// TODO write wrapper for converting between CMatrix and major armadillo/eigen types

// wrapper for eigen vector input x
double expected_improvement_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data) {
// double expected_improvement_optim(const ColVec_t& x, ColVec_t* grad_out, void* opt_data) {
    EIOptimStruct* optfn_data = reinterpret_cast<EIOptimStruct*>(opt_data);
    CMatrix x_cmat(1, (int)(x.rows()), x.data());
    return -expected_improvement(x_cmat, optfn_data->model, optfn_data->y_b, optfn_data->exp_bias);
}

struct GPOptimStruct {
    CGp model;
};

// double model_predict_optim(const ColVec_t& x, ColVec_t* grad_out, void* opt_data) {
double model_predict_optim(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data) {
    GPOptimStruct* optfn_data = reinterpret_cast<GPOptimStruct*>(opt_data);
    CMatrix x_cmat(1, (int)(x.rows()), x.data());
    CMatrix mu_mat(1, 1);
    CMatrix std_mat(1, 1);
    optfn_data->model.out(mu_mat, std_mat, x_cmat);
    return -(mu_mat(0, 0));
}
#endif  // _WIN


CMatrix* BayesianSearchModel::get_next_sample() {
    CMatrix* x = new CMatrix(1, x_dim);
    
    // initial random samples
    // TODO upgrade to Latin hypercube sampling
    if (num_samples < initial_samples) { uniform_random_sample(x); }
    else {
        try {
            // update Gaussian process model
            // TODO valid? want to free old gp memory if it was previously allocated
            if (gp != nullptr) { delete gp; }
            // want persistent access to model after fitting and getting next sample in order to fit new model

            // TODO move model set up to separate function, only set up function once after first sample
            if (noiseInit != nullptr) {delete noiseInit;}
            noiseInit = new CGaussianNoise(y_samples);
            for(int i=0; i<noiseInit->getOutputDim(); i++) {noiseInit->setBiasVal(0.0, i);}
            noiseInit->setParam(0.0, noiseInit->getOutputDim());
            
            int iters = 100;
            // bias is not actually learned currently (was not implemented in original CGp library)
            bool outputBiasScaleLearnt = false;
            int approxType = 0;

            gp = new CGp(kern, noiseInit, x_samples, approxType, -1, 3);
            gp->setBetaVal(1);
            gp->setScale(1.0);
            gp->setBias(0.0);
            gp->setObsNoiseVar(obsNoise);
            gp->updateM();

            gp->setVerbosity(verbosity);
            int default_optimiser = CGp::LBFGS_B;
            gp->setDefaultOptimiser(default_optimiser);  //options: LBFGS_B, BFGS, SCG, CG, GD

            gp->setObjectiveTol(1e-6);
            gp->setParamTol(1e-6);
            gp->setOutputBiasLearnt(outputBiasScaleLearnt);
            gp->setOutputScaleLearnt(outputBiasScaleLearnt);
            gp->set_seed(seed + 10000 + num_samples);
            gp->set_n_restarts(2);
            gp->pkern->setInitParam();
            gp->optimise(iters);

            // optimize acquisition function

            if (optimization_fcn.compare("grid") == 0) { x = new CMatrix(gridSearch(acq_fcn, grid_vals)); }
            #ifndef _WIN
            else if (optimization_fcn.compare("de") == 0) {
                Eigen::VectorXd x_optim = Eigen::VectorXd(x_dim);
                Eigen::VectorXd lower_bounds = Eigen::VectorXd(x_dim);
                Eigen::VectorXd upper_bounds = Eigen::VectorXd(x_dim);
                // ColVec_t x_optim = ColVec_t(x_dim);
                // ColVec_t lower_bounds = ColVec_t(x_dim);
                // ColVec_t upper_bounds = ColVec_t(x_dim);
                // TODO choose better/random initial values? not strictly needed for global optimization
                for (int i = 0; i < x_dim; i++) {
                    x_optim[i] = (bounds.getVal(i, 0) + bounds.getVal(i, 1))/2.0;
                    lower_bounds[i] = bounds.getVal(i, 0);
                    upper_bounds[i] = bounds.getVal(i, 1);
                }

                CMatrix x_test(1, x_dim, x_optim.data());
                if (verbosity>=2) {
                    double temp_EI = expected_improvement(x_test, *gp, y_best, exploration_bias);
                    cout << "Current acquisition function value: " << temp_EI << endl;
                }

                optim::algo_settings_t optim_settings;
                optim_settings.print_level = max(verbosity - 2, 0);
                optim_settings.vals_bound = true;
                // optim_settings.iter_max = 15;  // doesn't seem to control anything with DE
                optim_settings.rel_objfn_change_tol = 1e-05;
                optim_settings.rel_sol_change_tol = 1e-05;

                optim_settings.de_settings.n_pop = 25 * x_dim;
                optim_settings.de_settings.n_gen = 25 * x_dim;

                optim_settings.lower_bounds = lower_bounds;
                optim_settings.upper_bounds = upper_bounds;

                optim_settings.de_settings.initial_lb = lower_bounds;
                optim_settings.de_settings.initial_ub = upper_bounds;
                // latest version of optim has RNG seeding, but not current version used
                // optim_settings.rng_seed_value = seed;
                // seed eigen library instead with std library seeding
                srand((unsigned int) seed + num_samples);

                bool success;

                if (acq_func_name.compare("expected_improvement") == 0) {
                    EIOptimStruct opt_data = {*gp, y_best, exploration_bias};
                    success = optim::de(x_optim, expected_improvement_optim, &opt_data, optim_settings);
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
            #endif
            else { throw std::runtime_error(string("Unknown optimization function (optimization_fcn): ") + optimization_fcn); }
        }
        catch(const std::exception& e) {  // catch all errors in fitting and get random sample
            cout << "Warning: error in fitting process for getting next sample. Falling back on random parameter sampling." << endl;
            cout << "Error message: " << e.what() << endl;

            cout << "Rethrowing exception rather than handling with random resampling for debugging." << endl;
            throw e;
            
            uniform_random_sample(x);
        }
    }
    return x;
}


CMatrix* BayesianSearchModel::get_best_solution() {
    // optimize GP

    CMatrix* x = new CMatrix(1, x_dim);

    if (optimization_fcn.compare("grid") == 0) { x = new CMatrix(gridSearch(acq_fcn, grid_vals)); }
    #ifndef _WIN
    else if (optimization_fcn.compare("de") == 0) {
        x = new CMatrix(1, x_dim);

        Eigen::VectorXd x_optim = Eigen::VectorXd(x_dim);
        Eigen::VectorXd lower_bounds = Eigen::VectorXd(x_dim);
        Eigen::VectorXd upper_bounds = Eigen::VectorXd(x_dim);
        // ColVec_t x_optim = ColVec_t(x_dim);
        // ColVec_t lower_bounds = ColVec_t(x_dim);
        // ColVec_t upper_bounds = ColVec_t(x_dim);
        // TODO choose better/random initial values? not strictly needed for global optimization
        // TODO implement CMatrix.getCol/getRow
        for (int i = 0; i < x_dim; i++) {
            x_optim[i] = (bounds.getVal(i, 0) + bounds.getVal(i, 1))/2.0;
            lower_bounds[i] = bounds.getVal(i, 0);
            upper_bounds[i] = bounds.getVal(i, 1);
        }

        CMatrix x_test(1, x_dim, x_optim.data());

        optim::algo_settings_t optim_settings;
        // latest version of optim has RNG seeding, but not current version used
        // optim_settings.rng_seed_value = seed;
        // seed eigen library instead with std library seeding
        srand((unsigned int) seed - num_samples);
        optim_settings.print_level = verbosity;
        if (verbosity >= 1) optim_settings.print_level -= 1;
        optim_settings.vals_bound = true;
        // optim_settings.iter_max = 15;  // doesn't seem to control anything with DE
        optim_settings.rel_objfn_change_tol = 1e-05;
        optim_settings.rel_sol_change_tol = 1e-05;

        optim_settings.de_settings.n_pop = 100;
        optim_settings.de_settings.n_gen = 100;

        optim_settings.lower_bounds = lower_bounds;
        optim_settings.upper_bounds = upper_bounds;

        optim_settings.de_settings.initial_lb = lower_bounds;
        optim_settings.de_settings.initial_ub = upper_bounds;

        bool success;

        GPOptimStruct opt_data = {*gp};
        success = optim::de(x_optim, model_predict_optim, &opt_data, optim_settings);
        if (success) {
            for (int i = 0; i < x_dim; i++) {
                x->setVal(x_optim[i], i);
            }
        }
        else {
            // TODO exceptions aren't printing error messages in Visual Studio
            cout << "Optimization of GP prediction (mean) function failed." << endl;
            throw std::runtime_error("Optimization of GP prediction (mean) function failed.");
        }
    }
    #endif
    else { throw std::runtime_error(string("Unknown optimization function (optimization_fcn): ") + optimization_fcn); }
    if (verbosity >= 0) {
        CMatrix y(1, 1);
        gp->out(y, *x);
        cout << "CBayesianSearch: Best solution with " << num_samples << " samples: (x_best, y_pred): (";
        if (x_dim > 1) cout << "[";
        for (int i = 0; i < x_dim; i++) {
            cout  << x->getVal(i) << ", ";
        }
        if (x_dim > 1) cout << "]";
        cout << ", " << y.getVal(0) << ")" << endl;
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

    if (verbosity >= 1) {
        cout << "Sample " << num_samples << ": (x, y): (";
        if (x_dim > 1) cout << "[";
        for (int i = 0; i < x_dim; i++) {
            cout  << x.getVal(i) << ", ";
        }
        if (x_dim > 1) cout << "]";
        cout << ", " << y.getVal(0) << ")" << endl;
    }
}

void BayesianSearchModel::uniform_random_sample(CMatrix* x) {
    if (init_points_on_grid) {
        for (unsigned int i = 0; i < x_dim; i++) {
            std::uniform_int_distribution<> dist(0, grid_vals[i].getRows() - 1);
            // fix uniform random samples to grid points
            x->setVal(grid_vals[i].getVal(dist(rng)), i);
        }
    }
    else {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        for (unsigned int i = 0; i < x_dim; i++) {
            // fix uniform random samples to grid points
            x->setVal(bounds.getVal(i, 0) + dist(rng) * (bounds.getVal(i, 1) - bounds.getVal(i, 0)), i);
        }
    }
}
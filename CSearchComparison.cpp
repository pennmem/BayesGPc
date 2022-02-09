#include "CSearchComparison.h"

ComparisonStruct CSearchComparison::get_best_solution() {
    vector<CMatrix*> xs;
    vector<CMatrix> ys_pred;
    vector<double> mus;
    vector<double> vars;
    vector<double> ws;
    double w_sum = 0;
    double mu_total = 0;
    for (int i = 0; i < num_models; i++) {
        xs.push(models[i].get_best_solution());
        CMatrix y_pred(1, 1);
        CMatrix sem_pred(1, 1);
        CMatrix std_pred(1, 1);
        models[i].gp->out_sem(y_pred, sem_pred, xs[i]);
        ys_pred.push(y_pred);
        mus.push(y_pred.getVal(0));
        mu_total += mus[i];
        // reciprocated square-SEM
        ws.push(std::pow(1/sem_pred.getVal(0), 2));
        w_sum += ws[i];
        
        models[i].gp->out(y_pred, std_pred, xs[i]);
        vars += std::pow(std_pred.getVal(0), 2);
    }
    mu_total /= num_models;

    double MSTR = 0;
    for (int i = 0; i < num_models; i++) {
        MSTR += ws[i] * std::pow(mus[i] - mu_total, 2)
    }
    MSTR /= num_models - 1;
    
    double lambda = 0;
    for (int i = 0; i < num_models; i++) {
        // estimate effective number of local samples by comparing sem to total variance at x_best
        // TODO test this, problematic if noise level is not homogeneous over input space
        //      which it certainly isn't for amplitude (and likely duration and frequency as well)
        // likely problematic overall since white noise level along with other kernel hyperparameters
        // are somewhat unstable and highly dependent on initialization. I need to run tests of the
        // parameter stability and talk with Mike.
        double eff_n = vars[i] * ws[i];
        assert(eff_n > 1);
        lambda += std::pow(1.0 - ws[i]/w_sum, 2)/(eff_n - 1);
    }
    lambda *= 3.0 / (std::pow(num_models, 2) - 1);

    double F = MSTR/(1 + 2 * lambda * (num_models - 2)/3);
    
    double anova_pval = 1;
    vector<double> sham_pvals(num_models);

    ComparisonStruct res {x_best, xs, mus, anova_pval, sham_pvals, best_idx};
    return res;
}
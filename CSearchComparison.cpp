#include "CSearchComparison.h"

// TODO: break out Welch's ANOVA into separate function for unit testing
// compare predicted disributions from set of GPs (having optimal means for each GP) 
// using Welch's F-test/ANOVA
ComparisonStruct CSearchComparison::get_best_solution() {
    double best_val = -std::numeric_limits<double>::infinity();
    int idx_best = -1;
    vector<CMatrix*> xs;
    vector<double> mus;
    vector<double> sems;
    // effective GP sample sizes
    vector<double> eff_ns;

    // temp
    CMatrix y_pred(1, 1);
    CMatrix sem_pred(1, 1);
    CMatrix std_pred(1, 1);
    for (int i = 0; i < num_models; i++) {
        // get best predictions
        xs.push_back(models[i]->get_best_solution());
        models[i]->gp->out_sem(y_pred, sem_pred, *(xs[i]));
        mus.push_back(y_pred.getVal(0));
        sems.push_back(sem_pred.getVal(0));

        if (mus[i] > best_val) { idx_best = i; }
        
        // estimate effective GP sample sizes
        models[i]->gp->out(y_pred, std_pred, *(xs[i]));
        eff_ns.push_back(std::pow(std_pred.getVal(0) / sems[i], 2));
    }

    TestStruct test = anova_welch(mus, sems, eff_ns);

    // means not significantly different so choose most reliable model (smallest SEM)
    if (test.pval > pthreshold) {
        best_val = std::numeric_limits<double>::infinity();
        idx_best = -1;
        for (int i = 0; i < num_models; i++) {
            if (sems[i] < best_val) { idx_best = i; }
        }
    }

    ComparisonStruct res {idx_best, xs, mus, sems, eff_ns, test.pval};
    return res;
}


// compare distribution of samples to predicted distribution from GP using Welch's t-test
TestStruct CSearchComparison::compare_GP_to_sample(const ComparisonStruct& res, const vector<double>& dist_results) {
    double mu2 = 0;
    double sem2 = -1;
    double n2 = dist_results.size();
    for (int i = 0; i < n2; i++) { mu2 += dist_results[i]; }
    mu2 /= n2;
    for (int i = 0; i < n2; i++) { sem2 += std::pow(dist_results[i] - mu2, 2); }
    sem2 /= n2 - 1;  // variance estimate
    sem2 = std::sqrt(sem2/n2);
    TestStruct ttest_res = ttest_welch(res.mus[res.idx_best], mu2, 
                                       res.sems[res.idx_best], sem2, 
                                       res.ns[res.idx_best], n2);
    return ttest_res;
}


TestStruct anova_welch(vector<double> mus,
                       vector<double> sems,
                       vector<double> ns) {
    TestStruct res;
    int n_groups = mus.size();
    vector<double> ws;
    double w_sum = 0;
    double mu_total = 0;

    for (int i = 0; i < n_groups; i++) {
        // reciprocated square-SEM
        ws.push_back(std::pow(1/sems[i], 2));
        w_sum += ws[i];
        mu_total += ws[i] * mus[i];
    }
    mu_total /= w_sum;

    double MSTR = 0;
    for (int i = 0; i < n_groups; i++) {
        MSTR += ws[i] * std::pow(mus[i] - mu_total, 2);
    }
    MSTR /= n_groups - 1;
    
    double lambda = 0;
    for (int i = 0; i < n_groups; i++) {
        // TODO test this, problematic if noise level is not homogeneous over input space
        // estimate effective number of local samples by comparing sem to total variance at x_best
        //      which it certainly isn't for amplitude (and likely duration and frequency as well)
        //      may still not matter a ton
        // likely problematic overall since white noise level along with other kernel hyperparameters
        // are somewhat unstable and highly dependent on initialization. I need to run tests of the
        // parameter stability and talk with Mike.
        // better estimates of degrees of freedom available, see https://hastie.su.domains/Papers/cantoni_biometrika.pdf
        assert(ns[i] > 1);
        lambda += std::pow(1.0 - ws[i]/w_sum, 2)/(ns[i] - 1);
    }
    lambda *= 3.0 / (n_groups * n_groups - 1);

    res.stat = MSTR/(1 + 2 * lambda * (n_groups - 2)/3);
    boost::math::fisher_f Fdist(n_groups - 1, 1/lambda);
    
    res.pval = cdf(complement(Fdist, res.stat));
    return res;
}


// one-sided Welch's t-test for testing the alternative mu1 > mu2
TestStruct ttest_welch(double mu1, double mu2, 
                       double sem1, double sem2, 
                       double n1, double n2) {
    TestStruct res;

    double sem1_2 = sem1 * sem1;
    double sem2_2 = sem2 * sem2;

    // t-stat
    res.stat = (mu1 - mu2)/std::sqrt(sem1_2 + sem2_2);
    
    double dof = std::pow(sem1_2 + sem2_2, 2) / (std::pow(sem1_2, 2)/(n1 - 1) + std::pow(sem2_2, 2)/(n2 - 1));
    boost::math::students_t dist(dof);
    res.pval = cdf(complement(dist, res.stat));

    return res;
}


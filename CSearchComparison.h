#ifndef CSEARCHCOMPARISON_H
#define CSEARCHCOMPARISON_H

#include "CMatrix.h"
#include "CBayesianSearch.h"
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>

struct ComparisonStruct {
    int idx_best;
    vector<CMatrix*> xs;
    vector<double> mus;
    vector<double> sems;
    vector<double> ns;  // effective sample sizes
    double pval;
};

struct TestStruct {
    double stat;
    double pval;
};

class CSearchComparison {
    public:
    CSearchComparison() {}
    CSearchComparison(int n_models, double alpha, vector<CCmpndKern> kernels, vector<CMatrix> param_bounds,
            vector<double> observation_noises, vector<double> exploration_biases, vector<int> init_samples,
            vector<int> rng_seeds, int verbose) {
        num_models = n_models;
        pthreshold = alpha;
        kerns = kernels;
        param_bounds = param_bounds;
        obsNoises = observation_noises;
        exp_biases = exploration_biases;
        initial_samples = init_samples;
        seeds = rng_seeds;
        verbosity = verbose;

        for (int i = 0; i < num_models; i++) {
            models.push_back(new BayesianSearchModel(kerns[i], param_bounds[i],
                    obsNoises[i], exp_biases[i],
                    initial_samples[i], seeds[i], verbosity));
        }
    }
    ~CSearchComparison() {
        for (int i = 0; i < num_models; i++) {
            BayesianSearchModel* temp = models.back();
            models.pop_back();
            delete temp;
        }
    }

    CMatrix* get_next_sample(unsigned int model_idx);
    void add_sample(unsigned int model_idx, const CMatrix& x, const CMatrix& y);
    ComparisonStruct get_best_solution();
    TestStruct compare_GP_to_sample(const ComparisonStruct& res, const vector<double>& dist_results);

    unsigned int num_models;
    double pthreshold;
    vector<BayesianSearchModel*> models;
    vector<CCmpndKern> kerns;
    vector<CMatrix> param_bounds;
    vector<double> obsNoises;
    vector<double> exp_biases;
    vector<int> initial_samples;
    vector<int> seeds;
    int verbosity;
};

TestStruct anova_welch(vector<double> mus, vector<double> sems, vector<double> ns);
// one-sided Welch's t-test for testing the alternative mu1 > mu2
TestStruct ttest_welch(double mu1, double mu2, double sem1, double sem2, double n1, double n2);

#endif

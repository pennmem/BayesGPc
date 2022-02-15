#include "CMatrix.h"
#include "CBayesianSearch.h"
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>

struct ComparisonStruct {
    int idx_best,
    vector<CMatrix> xs,
    vector<double> mus,
    vector<double> sems,
    vector<double> ns,  // effective sample sizes
    double pval,
};

struct TestStruct {
    double stat,
    double pval
}

class CSearchComparison {
    CSearchComparison() {}

    CSearchComparison(int n_models, double alpha, vector<CCmpndKern> kernels, vector<CMatrix> param_bounds, 
            vector<double> observation_noises, vector<double> exp_biases, vector<int> init_samples, 
            vector<int> rng_seeds, int verbose) {
        num_models = n_models;
        pthreshold = alpha;
        kerns = kernels;
        bounds = param_bounds;
        obsNoises = observation_noises; 
        exploration_biases = exp_biases;
        initial_samples = init_samples; 
        seeds = rng_seed;
        verbosity = verbose;

        for (int i = 0; i < num_models; i++) {
            models.push(BayesianSearchModel(kerns[i], bounds[i]*, 
                    obsNoise[i], exploration_biases[i], 
                    initial_samples[i], seeds[i], verbosity));
        }
    }

    CMatrix* get_next_sample(unsigned int model_idx);
    void add_sample(unsigned int model_idx, const CMatrix& x, const CMatrix& y);
    ComparisonStruct get_best_solution();

    unsigned int num_models;
    double pthreshold;
    vector<BayesianSearchModel> models;
    vector<CCmpndKern> kerns;
    vector<CMatrix> param_bounds;
    vector<double> obsNoises;
    vector<double> exp_biases;
    vector<int> initial_samples;
    vector<int> seeds;
    int verbosity;
}


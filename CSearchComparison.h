#include "CMatrix.h"
#include "CBayesianSearch.h"

struct ComparisonStruct {
    CMatrix x,
    vector<CMatrix> xs,
    vector<double> mus,
    double anova_pval,
    vector<double> sham_pvals
    int model_idx,
};

class CSearchComparison {
    CSearchComparison() {}

    CSearchComparison(int n_models, vector<CCmpndKern> kernels, vector<CMatrix> param_bounds, 
            vector<double> observation_noises, vector<double> exp_biases, vector<int> init_samples, 
            vector<int> rng_seeds, int verbose) {
        num_models = n_models;
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
    vector<BayesianSearchModel> models;
    vector<CCmpndKern> kerns;
    vector<CMatrix> param_bounds;
    vector<double> obsNoises;
    vector<double> exp_biases;
    vector<int> initial_samples;
    vector<int> seeds;
    int verbosity;
}


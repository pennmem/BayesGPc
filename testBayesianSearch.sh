# test Bayesian search implementation with main test functions

# $ nohup $(testBayesianSearch.sh) &

kernels=("Matern32" "RBF")

noise_levels=(0.0 ) # 0.1 0.3)
exp_biases=(0.1 ) # 0.25)
init_samples=(25 ) # 100)
for k in "${kernels[@]}"
do
for n in "${noise_levels[@]}"
do
for e in "${exp_biases[@]}"
do
for s in "${init_samples[@]}"
do
    ./testBayesianSearch --tag run --func sin --noise_level $n --exp_bias $e --init_samples $s --n_runs 2 --kernel $k
done
done
done
done

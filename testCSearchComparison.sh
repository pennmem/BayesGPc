#!/bin/bash

# test Bayesian search comparison implementation with main test functions
# $ nohup $(testCSearchComparison.sh) [TAG] [IMPL] &
# TAG: tag for test run, IMPL: implementation to test.

if [[ "$#" -lt 1 ]]; then
    TAG="test"
    IMPL="ANOVA"
elif [[ "$#" -lt 2 ]]; then
    TAG=$1
    IMPL="ANOVA"
else
    TAG=$1
    IMPL=$2
fi

# shift 2

SMOKESCREEN=0
# if [[ "$#" -lt 1 ]]; then
#     SMOKESCREEN=1
# else
#     SMOKESCREEN=0
# fi

if [ $IMPL != "ANOVA" ]; then
    echo "Implementation '${IMPL}' not implemented. Only 'ANOVA' currently supported. Exiting."
    return 1
fi

if [[ $SMOKESCREEN -eq 1 ]]; then
    LOGDIR=$(pwd)/results/debug/${TAG}_${IMPL}
else
    LOGDIR=$(pwd)/results/${TAG}_${IMPL}
fi

if test -d "${LOGDIR}"; then
    echo "Experiment tag '${TAG}' already used. Exiting."
    return 1
else 
    echo "Log directory: ${LOGDIR}"
fi

mkdir $LOGDIR
ARGS_FILE=$LOGDIR/args.txt
touch ARGS_FILE

if [[ $SMOKESCREEN -eq 1 ]]; then
    n_iters=(27)
    n_grids=(5)
    n_runs=2
    kernels=("Matern32")
    func="schwefel"
    noise_levels=(0.0)
    init_samples=(25)
    exp_biases=(0.1)
    n_ways=(2)
    mean_diffs=(0.3)
else
    n_iters=(150 250)
    n_grids=(5 30)
    n_runs=50
    kernels=("Matern32")
    func="all"
    noise_levels=(0.1 0.3 0.4)
    exp_biases=(0.1 0.5)
    n_ways=(2 6 10 20)
    # 0.0 0.1 0.3 0.5 1.0 2.0)
    mean_diffs=(0.0)
    # 25 100
    init_samples=(100)  # 100 in Nia implementation
fi

for n in "${noise_levels[@]}"
do
for ng in "${n_grids[@]}"
do
for m in "${mean_diffs[@]}"
do
for w in "${n_ways[@]}"
do
for si in "${n_iters[@]}"
do
for s in "${init_samples[@]}"
do
for k in "${kernels[@]}"
do
for e in "${exp_biases[@]}"
do
    args="--tag ${TAG} --func ${func} --noise_level ${n} --exp_bias ${e} --n_init_samples ${s} --n_runs ${n_runs} --kernel ${k} --n_iters ${si} --n_way ${w} --mean_diff ${m} --n_grid ${ng}"
    if [ $IMPL == "skopt" ]; then
        args="--impl ${IMPL} ${args}"
    fi
    echo $args >> $ARGS_FILE
done
done
done
done
done
done
done
done

curdir=$(pwd)
echo "pwd: ${curdir}"
if [ $IMPL == "ANOVA" ]; then
    source parallel_exec.sh ${curdir}/testCSearchComparison_full $ARGS_FILE $LOGDIR testCSearchComparison_full
fi

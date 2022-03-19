#!/bin/bash

# test Bayesian search implementation with main test functions
# $ nohup $(testBayesianSearch.sh) [TAG] [IMPL] &
# TAG: tag for test run, IMPL: implementation to test.

if [[ "$#" -lt 1 ]]; then
    TAG="test"
    IMPL="CBay"
elif [[ "$#" -lt 2 ]]; then
    TAG=$1
    IMPL="CBay"
else
    TAG=$1
    IMPL=$2
fi

# shift 2

SMOKESCREEN=1
# if [[ "$#" -lt 1 ]]; then
#     SMOKESCREEN=1
# else
#     SMOKESCREEN=0
# fi

if [ $IMPL != "CBay" -a $IMPL != "skopt" -a $IMPL != "nia"]; then
    echo "Implementation '${IMPL}' not implemented. Only 'CBay', 'skopt', and 'nia' currently supported. Exiting."
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
    n_runs=2
    kernels=("Matern32")
    func="schwefel"
    noise_levels=(0.0)
    init_samples=(25)
    exp_biases=(0.1)
else
    n_iters=(150, 250)
    n_runs=50
    kernels=("Matern32")  # "Matern52" "RBF" "RationalQuadratic")
    func="all"
    noise_levels=(0.0 0.1 0.3)
    exp_biases=(0.0 0.25 0.5 1.0 2.0)
    init_samples=(25 100)  # 100 in Nia implementation
fi

for n in "${noise_levels[@]}"
do
for niter in "${n_iters[@]}"
do
for s in "${init_samples[@]}"
do
    if [ $IMPL == "nia" ]; then
        args="--tag ${TAG} --func ${func} --noise_level ${n} --n_init_samples ${s} --n_iters ${niter} --n_runs ${n_runs}"
        args="--impl ${IMPL} ${args}"
        echo $args >> $ARGS_FILE
        continue
    fi

    for k in "${kernels[@]}"
    do
    for e in "${exp_biases[@]}"
    do
        args="--tag ${TAG} --func ${func} --noise_level ${n} --exp_bias ${e} --n_init_samples ${s} --n_runs ${n_runs} --kernel ${k} --n_iters ${niter}"
        if [ $IMPL == "skopt" ]; then
            args="--impl ${IMPL} ${args}"
        fi
        echo $args >> $ARGS_FILE
    done
    done
done
done
done

curdir=$(pwd)
echo "pwd: ${curdir}"
if [ $IMPL == "CBay" ]; then
    source parallel_exec.sh ${curdir}/testBayesianSearch $ARGS_FILE $LOGDIR testBayesianSearch
else
    source parallel_exec.sh ${curdir}/ReferenceBayesianSearch.py $ARGS_FILE $LOGDIR testBayesianSearch_py
fi

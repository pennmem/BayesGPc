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

# shift 1

# if [[ "$#" -lt 1 ]]; then
#     SMOKESCREEN=1
# else
#     SMOKESCREEN=0
# fi

if [ $IMPL != "CBay" -a $IMPL != "skopt" ]; then
    echo "Implementation '${IMPL}' not implemented. Only 'CBay' and 'skopt' currently supported. Exiting."
    return 1
fi

LOGDIR=$(pwd)/results/${TAG}_${IMPL}
if test -d "${LOGDIR}"; then
    echo "Experiment tag '${TAG}' already used. Exiting."
    return 1
else 
    echo "Log directory: ${LOGDIR}"
fi

mkdir $LOGDIR
ARGS_FILE=$LOGDIR/args.txt
touch ARGS_FILE

# arguments
kernels=("Matern32")
# kernels=("Matern32" "RBF")
noise_levels=(0.0)
# 0.1 0.3)
exp_biases=(0.1)
# 0.25 0.4)
init_samples=(25)
# 100)

for k in "${kernels[@]}"
do
for n in "${noise_levels[@]}"
do
for e in "${exp_biases[@]}"
do
for s in "${init_samples[@]}"
do
    args="--tag ${TAG} --func all --noise_level ${n} --exp_bias ${e} --n_init_samples ${s} --n_runs 2 --kern ${k} --n_iters 150"
    # nohup ./testBayesianSearch $args &
    if [ $IMPL != "CBay" ]; then
        args="--impl ${IMPL} ${args}"
    fi

    echo $args >> $ARGS_FILE
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

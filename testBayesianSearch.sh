#!/bin/bash

# test Bayesian search implementation with main test functions
# $ nohup $(testBayesianSearch.sh) &

if [[ "$#" -lt 1 ]]; then
    TAG="test"
else
    TAG=$1
fi

# shift 1

# if [[ "$#" -lt 1 ]]; then
#     SMOKESCREEN=1
# else
#     SMOKESCREEN=0
# fi

LOGDIR=$(pwd)/results/${TAG}
if test -d "${LOGDIR}"; then
    echo "Experiment tag '${TAG}' already used. Exiting."
    exit 1
else 
    echo "Log directory: ${LOGDIR}"
fi

mkdir $LOGDIR
ARGS_FILE=$LOGDIR/args.txt
touch ARGS_FILE

# arguments
kernels=("Matern32" "RBF")
noise_levels=(0.0 0.1 0.3)
exp_biases=(0.1 0.25 0.4)
init_samples=(25 100)

for k in "${kernels[@]}"
do
for n in "${noise_levels[@]}"
do
for e in "${exp_biases[@]}"
do
for s in "${init_samples[@]}"
do
    args="--tag ${TAG} --func all --noise_level ${n} --exp_bias ${e} --init_samples ${s} --n_runs 25 --kernel ${k}"
    # nohup ./testBayesianSearch $args &
    echo $args >> $ARGS_FILE
done
done
done
done

curdir=$(pwd)
echo "pwd: ${curdir}"
source parallel_exec.sh ${curdir}/testBayesianSearch $ARGS_FILE $LOGDIR testBayesianSearch

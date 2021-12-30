if [[ "$#" -lt 2 ]]; then
    echo "USAGE: parallel.sh [executable] [arguments_file] [logdir (optional)] [cluster job name (optional)]"
    exit 1
fi

EXEC=$1
ARGS_FILE="$2"
NUM_JOBS=`wc -l < "$ARGS_FILE"`
shift 2
if [[ "$#" -lt 1 ]]; then
    LOGDIR="$HOME/logs/${EXEC}"
else
    LOGDIR="$1"
    shift
fi

if [[ "$#" -lt 1 ]]; then
    JOB_NAME="parallel_exec_job"
else
    JOB_NAME="$1"
    shift
fi

echo $ARGS_FILE
echo $LOGDIR

echo "Running ${EXEC}"
echo "Reading arguments from ${ARGS_FILE}"
echo "Running ${NUM_JOBS} jobs"
echo "Outputting to ${LOGDIR}"
echo "Job name: ${JOB_NAME}"
echo ""

while IFS=" " ; read -r arr
    do
        sleep 5
        echo "$arr"
        qsub -b y -N $JOB_NAME -q "RAM.q" -l h_vmem=90G,s_vmem=90G -e "${LOGDIR}/err.${arr// /_}" -o "${LOGDIR}/out.${arr// /_}" $EXEC ${arr} --logdir $LOGDIR
done < "${ARGS_FILE}"

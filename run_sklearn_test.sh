echo $#
if [ $# -gt 0 ] ; then
	#python testSklearn.py
	make testGp_sklearn
	./testGp_sklearn -k $1
	make gp
	data_file="np_files/testSklearn_gpr_$1.npz"
	#printf "$data_file\n"
	./gp gnuplot -r 400 $data_file testSklearn.model testSklearn
	gnuplot testSklearn_plot.gp
else
	echo "Must specify kernel name, e.g., 'matern32'"
fi


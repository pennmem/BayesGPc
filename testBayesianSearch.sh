# main test functions across sample noise levels
nohup ./testBayesianSearch --noise_level 0.1 --exp_bias 0.1 --init_samples 25 > ./results/testAll_initial_lfbgs_bounded_noise01_exp01_init_samps25_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.1 --exp_bias 0.25 --init_samples 25 > ./results/testAll_initial_lfbgs_bounded_noise01_exp025_init_samps25_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.1 --exp_bias 0.1 --init_samples 100 > ./results/testAll_initial_lfbgs_bounded_noise01_exp01_init_samps100_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.1 --exp_bias 0.25 --init_samples 100 > ./results/testAll_initial_lfbgs_bounded_noise01_exp025_init_samps100_fixed_obs_noise_exp_bias.txt 2>&1 &


nohup ./testBayesianSearch --noise_level 0.2 --exp_bias 0.1 --init_samples 25 > ./results/testAll_initial_lfbgs_bounded_noise02_exp01_init_samps25_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.2 --exp_bias 0.25 --init_samples 25 > ./results/testAll_initial_lfbgs_bounded_noise02_exp025_init_samps25_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.2 --exp_bias 0.1 --init_samples 100 > ./results/testAll_initial_lfbgs_bounded_noise02_exp01_init_samps100_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.2 --exp_bias 0.25 --init_samples 100 > ./results/testAll_initial_lfbgs_bounded_noise02_exp025_init_samps100_fixed_obs_noise_exp_bias.txt 2>&1 &


nohup ./testBayesianSearch --noise_level 0.3 --exp_bias 0.1 --init_samples 25 > ./results/testAll_initial_lfbgs_bounded_noise03_exp01_init_samps25_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.3 --exp_bias 0.25 --init_samples 25 > ./results/testAll_initial_lfbgs_bounded_noise03_exp025_init_samps25_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.3 --exp_bias 0.1 --init_samples 100 > ./results/testAll_initial_lfbgs_bounded_noise03_exp01_init_samps100_fixed_obs_noise_exp_bias.txt 2>&1 &

nohup ./testBayesianSearch --noise_level 0.3 --exp_bias 0.25 --init_samples 100 > ./results/testAll_initial_lfbgs_bounded_noise03_exp025_init_samps100_fixed_obs_noise_exp_bias.txt 2>&1 &

#! /bin/bash 
#
#SBATCH --output=/pscratch/sd/a/athomsen/run_files/v14/esub_logs/job_main_r_index%a.o 
#SBATCH --error=/pscratch/sd/a/athomsen/run_files/v14/esub_logs/job_main_r_index%a.e 
#SBATCH --job-name=job_main 
#SBATCH --account=des 
#SBATCH --constraint=cpu 
#SBATCH --qos=shared 
#SBATCH --licenses=cfs 
#SBATCH --licenses=scratch 
#SBATCH --array=1-3 
#SBATCH --time=0:12:00 
#SBATCH --mem-per-cpu=1952 
#SBATCH --tmp=0 
#SBATCH --cpus-per-task=2 

srun bash;  python -m esub.submit --job_name=job --tasks='55,332,492' --source_file=../../pipelines/v14/perlmutter_setup.sh --main_memory=1952.0 --main_time=0.2 --main_scratch=0.0 --function=main --executable=/global/u2/a/athomsen/multiprobe-simulation-inference/msi/apps/run_mcmc_for_coverage_tests.py --n_jobs=3 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs --system=slurm --main_name=main --batchsize=100000 --max_njobs=-1 --main_n_cores=2 --main_gpu=0 --main_gpu_memory=1000.0 --esub_verbosity=3 --main_mode=jobarray --mode=jobarray --additional_bsub_args='' --additional_slurm_args='--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch'  --keep_submit_files --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_test_v4/likelihood_flow --n_sims=1000 --jobname=mcmc_test
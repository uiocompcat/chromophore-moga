"#!/bin/bash -l
#SBATCH --account=NN4654K
#SBATCH --job-name=<job_name>
#SBATCH --time=<time_limit>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=<ntasks_per_node>
#SBATCH --mem=32G
#SBATCH --output=slurm.%j.log

#echo $SLURM_ARRAY_TASK_ID

args=("$@")
input=${args[$SLURM_ARRAY_TASK_ID]}
echo $input

# make the program and environment visible to this script
module --quiet purge
module load Gaussian/g16_C.01

export GAUSS_LFLAGS2="--LindaOptions -s 20000000"
export PGI_FASTMATH_CPU=avx2

# name of input file without extension
input=${args[$SLURM_ARRAY_TASK_ID]}

# create the temporary folder
export GAUSS_SCRDIR=/cluster/work/users/$USER/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p $GAUSS_SCRDIR

# copy input file to temporary folder
cp $SLURM_SUBMIT_DIR/$input.com $GAUSS_SCRDIR

# run the program
cd $GAUSS_SCRDIR
time g16.ib $input.com > $input.out

# copy result files back to submit directory
cp $input.out $SLURM_SUBMIT_DIR

exit 0

#!/bin/bash

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/output_%x-%A.%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/output_%x-%A.%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH -c 8
#SBATCH --mail-type=END,FAIL

echo "Workingdir: $PWD";
echo "Started at $(date)";

[ -d logs ] || mkdir logs

LOG_DIR="$PWD/logs"

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/miniconda3/bin/activate base # Adjust to your path of Miniconda installation
conda activate naslib
# Running the job
start=`date +%s`

BASE_DIR="."

bash $BASE_DIR/predictors/run_nb201_imagenet.sh $1 $BASE_DIR/saved_features/nasbench201-ImageNet16-120.pickle \
     $BASE_DIR/../../../zc_combine/data/nb201_valid_nets.csv $2 $3

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime


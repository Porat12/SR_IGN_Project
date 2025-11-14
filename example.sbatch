#!/bin/bash
#
# SBATCH Submission Script: PyTorch Job
#

# --- Job Configuration ---
#SBATCH --job-name=simulation_1       # Descriptive name for tracking the job.
#SBATCH --output=slurm-%j.out         # Redirects standard output to a file named with the job ID.
#SBATCH --error=slurm-%j.err          # Redirects standard error to a file.
#SBATCH --nodes=1                     # Confines the job to a single node for efficient GPU communication.
#SBATCH --mem=2G
#SBATCH --ntasks=1                    # Runs the entire job as a single task.
#SBATCH --time=02:00:00
# --- Mail notification ---
#SBATCH --mail-type=ALL               # Valid notification types: NONE, BEGIN, END, FAIL, REQUEUE, ALL, INVALID_DEPEND, STAGE_OUT, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50 and ARRAY_TASKS
#SBATCH --mail-user=porat.hai@campus.technion.ac.il

# --- Resource Allocation ---
#SBATCH --partition=l40s-shared       # Submits the job to the dedicated L40S GPU  partition or the “-l40s-shared” if you do not mind being Preempted on nodes which are not public.
#SBATCH --qos=2h_2g                  # Assigns the job to the public '24h_4g' Quality of Service tier for running jobs on public nodes partition “l40s-public”.
#SBATCH --gres=gpu:nvidia_l40s:1      
#SBATCH --cpus-per-gpu=8             # Allocates 12 CPU cores per requested GPU, ensuring no I/O bottleneck.

# --- Container & Mounts ---
#SBATCH --container=/usr/local/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif            # Specifies the Apptainer image to use.

# When using "--container", HOME, /scratch and Current-Working-Directory are mounted by default inside the container.
# Add additional bind mounts (delimited with a comma ,) by defining the mounts in the APPTAINER_BIND environment variable:
# export APPTAINER_BIND="$PRJ_WORKSPACE"
# sbatch <this_batch_script>
# OR:
# export APPTAINER_BIND="$PRJ_WORKSPACE,/home/maytalc/some/other/file.txt:/mount/inside/the/container/file.txt"
# sbatch <this_batch_script>

# --- Execution Block ---
set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID"
echo "Apptainer Container  : $APPTAINER_CONTAINER"
echo "Container Mounts     : $APPTAINER_BIND"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "================================================================"

echo ""
echo "--- Verifying GPU Accessibility Inside Container ---"
nvidia-smi
echo ""

df -h

echo "--- Launching Primary Command ---"
#
# --- Launching Primary Command ---

echo "Changing directory to project folder..."
cd /rg/shocher_prj/porat.hai/SR_IGN_Project/utils/download_datasets

echo "Current directory: $(pwd)"
echo "Listing files: $(ls)"

echo "Starting python script..."
# Run training script as a single command (all args on one line for clarity)
# python main.py --dataset_name celebA --batch_size 256 --HR_img_size 64 --scale_factor 2 \
#     --learning_rate 0.001 --beta_1 0.5 --beta_2 0.999 --weight_decay 0.0001 \
#     --lam_rec 15 --lam_idem 10 --lam_tight 5 --lam_SR 15 --a 3 \
#     --epochs 2

python download_DIV2K.py
echo "Mission Complete."

#!/bin/sh
#SBATCH -c 12
#SBATCH --ntask=1
#SBATCH --mem-per-cpu 15000
#SBATCH --time=03-00:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1-81
#SBATCH --output=om_result_train_KeRNL_v1_MNIST_%a.out

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/simg_images
singularity exec -B /om:/om /om/user/`whoami`/simg_images/tensorflow-1.12.0-py3.simg /usr/bin/python ~/MyCodes/KeRNL/keRNL_v1_MNIST_train_with_args_test_run.py $SLURM_ARRAY_TASK_ID

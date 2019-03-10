#!/bin/sh
#SBATCH -c 20
#SBATCH --ntask=1
#SBATCH --mem-per-cpu 10000
#SBATCH --time=03-00:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1
#SBATCH --output=om_result_train_KeRNL_v2_VS_BPTT_MNIST_%a.out

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/simg_images
singularity exec -B /om:/om /om/user/`whoami`/simg_images/tensorflow-1.12.0-py3.simg /usr/bin/python ~/MyCodes/KeRNL/om_train_keRNL_v2_VS_BPTT_MNIST_with_args.py $SLURM_ARRAY_TASK_ID

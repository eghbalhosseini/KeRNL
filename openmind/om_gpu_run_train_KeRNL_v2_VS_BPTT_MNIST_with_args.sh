#!/bin/sh
#SBATCH --job-name=om_train_KeRNL
#SBATCH --mem=20000
#SBATCH --gres= tesla-k20:1
#SBATCH --time=60:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1
#SBATCH --output=om_result_gpu_train_KeRNL_v2_VS_BPTT_MNIST_%a.out

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/simg_images
singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/tensorflow-latest-gpu-py3.simg /usr/bin/python ~/MyCodes/KeRNL/om_train_keRNL_v2_VS_BPTT_MNIST_with_args.py $SLURM_ARRAY_TASK_ID

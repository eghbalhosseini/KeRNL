#!/bin/sh
#SBATCH --job-name=om_train_KeRNL
#SBATCH --mem=20000
#SBATCH --gres=gpu:titan-x:2
#SBATCH --time=0-05:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1
#SBATCH --output=om_gpu_train_results_KeRNL_v2_VS_BPTT_MNIST_%a.out

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
singularity exec --nv -B /om:/om /om/user/`whoami`/st/om_image_tensorflow1-12_gpu.simg /usr/bin/python3 ~/MyCodes/KeRNL/om_train_keRNL_v2_VS_BPTT_MNIST_with_args.py $SLURM_ARRAY_TASK_ID

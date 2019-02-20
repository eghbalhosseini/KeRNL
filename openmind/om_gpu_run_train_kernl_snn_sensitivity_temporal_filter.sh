#!/bin/sh
#SBATCH --job-name=om_train_KeRNL
#SBATCH --mem=40000
#SBATCH --gres=gpu:titan-x:2
#SBATCH --time=1-00:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1
#SBATCH --output=om_gpu_train_results_kernel_train_sensitivity_and_temporal_filter.out

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
singularity exec --nv -B /om:/om /om/user/`whoami`/st/om_image_tensorflow1-12_gpu.simg /usr/bin/python3 ~/MyCodes/KeRNL/om_train_gpu_kernl_snn_sensitivity_tensor_and_filter.py

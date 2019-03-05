#!/bin/sh
#SBATCH --job-name=kernl_snn
#SBATCH --mem=40000
#SBATCH --gres=gpu:titan-x:2
#SBATCH --time=1-00:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1
#SBATCH --output=om_gpu_train_results_kernl_snn_adding_dataset.out

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om2/user/`whoami`/st/
singularity exec --nv -B /om2:/om2 /om2/user/`whoami`/simg_images/om_image_tensorflow1-12_gpu.simg /usr/bin/python3 ~/MyCodes/KeRNL/om_train_gpu_kernl_snn_adding_dataset.py

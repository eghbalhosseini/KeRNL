#!/bin/sh
#SBATCH --job-name=Kernl_ffn
#SBATCH --mem=40000
#SBATCH --gres=gpu:titan-x:1
#SBATCH --time=0-02:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --array=1
#SBATCH --output=om_gpu_train_results_kernl_ffn_mnist_%j.out
#SBATCH --error=om_gpu_train_results_kernl_ffn_mnist_%j.err

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/om_image_tensorflow1-12_gpu.simg /usr/bin/python3 ~/MyCodes/KeRNL/om_train_gpu_kernl_ffn_mnist.py

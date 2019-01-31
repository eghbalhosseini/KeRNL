#!/bin/sh
#SBATCH --output=om_result_train_KeRNL_v1_MNIST.out
#SBATCH -c 4
#SBATCH --ntask=1
#SBATCH --mem-per-cpu 15000
#SBATCH --time=00-01:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/simg_images
singularity exec -B /om:/om /om/user/`whoami`/simg_images/tensorflow-1.12.0-py3.simg /usr/bin/python ~/MyCodes/KeRNL/om_train_keRNL_v1_MNIST.py

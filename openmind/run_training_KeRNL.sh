#!/bin/sh
#SBATCH --output=run_training_KeRNL.out
#SBATCH -c 12
#SBATCH --ntask=1
#SBATCH --mem-per-cpu 15000
#SBATCH --time=00-01:00:00
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

#singularity shell docker://tensorflow/tensorflow:1.12.0-py3

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/simg_images
singularity exec -B /om:/om /om/user/`whoami`/simg_images/tensorflow-1.12.0-py3.simg /usr/bin/python -c "import tensorflow as tf; print(tf.__version__)"

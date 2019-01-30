#!/bin/sh
#SBATCH --output=Reconstruct3D_AllFrames_Vertical_CPU_V2.out
#SBATCH -c 12
#SBATCH --ntask=1
#SBATCH --mem-per-cpu 15000
#SBATCH --time=05-00:00:00
#SBATCH --array=1-500%100
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


module add mit/matlab/2017a
matlab -nodisplay -signelCompThread -r "addpath(genpath('~/MyCodes/DualLensLightField/')); \
recordingDay='12_15_17_40X_worm';\
VideoNumber='video 1';\
Direction='horizontal';\
data_folder=['/om/project/boyden/DualLensLightField/',recordingDay];cd(data_folder); \
InputPath=[data_folder,'/',VideoNumber,'/',Direction];\
PSFPath=[data_folder,'/psf'];\
FirstFramePath=[InputPath,'/Reconstructed'];\
SavePath=[data_folder,'/',VideoNumber,'/',Direction];\
Reconstruction3D_script_Frame2ToEnd(InputPath,PSFPath,SavePath,$SLURM_ARRAY_TASK_ID,FirstFramePath);\
quit;"

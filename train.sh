#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J EfficientDet
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u fets@elektro.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train.out
#BSUB -e train.err
# -- end of LSF options --


# Load the cuda module


module load python3/3.7.5
module load cudnn/v7.6.5.32-prod-cuda-10.0
source /work3/fets/git/EfficientDet.Pytorch/venv/bin/activate
#export PATH="$PATH:/zhome/fb/6/88845/.local/bin"

python train.py --dataset CSV --dataset_root  /work3/fets/mmdet/2019_04_12.csv --classes classes.txt --valset_root /work3/fets/mmdet/tracking_test_rgb.csv --network efficientdet-d4 --batch_size 8 --num_class 2
#python train.py --dataset CSV --dataset_root  /work3/fets/mmdet/tracking_test_rgb.csv --classes classes.txt --valset_root /work3/fets/mmdet/tracking_test_rgb.csv --network efficientdet-d4 --batch_size 8 --num_class 2
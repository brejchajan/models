#!/bin/bash

#$ -S /bin/bash
#$ -e /mnt/matylda1/ibrejcha/devel/tensorflow_models/research/delf/delf/logs
#$ -o /mnt/matylda1/ibrejcha/devel/tensorflow_models/research/delf/delf/logs
#$ -q long.q@@gpu
#$ -l ram_free=10G,mem_free=10G,matylda1=1,gpu=1,gpu_ram=8G,h=!pcgpu4

PROJECT_DIR="/mnt/matylda1/ibrejcha/devel/tensorflow_models/research/delf/delf/python"
cd $PROJECT_DIR || exit

source venv/bin/activate

gpu=$(nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p")
if [ "$gpu" == "" ]; then
	echo "No free GPU!"
    exit 1
fi
LD_LIBRARY_PATH="/homes/kazi/ibrejcha/lib64:/usr/local/share/cuda-10.2/lib64/:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=$gpu python3 training/train.py --train_file_pattern=/mnt/matylda1/ibrejcha/data/delg/Alps_photo_depth_query_resolution_depth_scaled_1000/tfrecord_nonorm/train* --validation_file_pattern=/mnt/matylda1/ibrejcha/data/delg/Alps_photo_depth_query_resolution_depth_scaled_1000/tfrecord_nonorm/validation* --imagenet_checkpoint=training/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --logdir=/mnt/matylda1/ibrejcha/data/delg/Alps_photo_depth_query_resolution_depth_scaled_1000/training_nonorm_autoencoder_numclasses --delg_global_features --dataset_version=alps

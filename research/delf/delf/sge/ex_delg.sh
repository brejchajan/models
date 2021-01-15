#!/bin/bash

#$ -S /bin/bash
#$ -e /mnt/matylda1/ibrejcha/devel/tensorflow_models/research/delf/delf/logs
#$ -o /mnt/matylda1/ibrejcha/devel/tensorflow_models/research/delf/delf/logs
#$ -q long.q@@gpu
#$ -l ram_free=10G,mem_free=10G,matylda1=1,gpu=1,gpu_ram=8G,h=!pcgpu4

PROJECT_DIR="/mnt/matylda1/ibrejcha/devel/tensorflow_models/research/delf/delf/python/examples"
cd $PROJECT_DIR || exit

source ../venv/bin/activate

gpu=$(nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p")
if [ "$gpu" == "" ]; then
	echo "No free GPU!"
    exit 1
fi
cmd="python3 extract_features_delg.py --config_path r50delg_gldv2clean_config.pbtxt --list_images_path switzerland_query_scale.txt --output_dir /mnt/matylda1/ibrejcha/data/delg/Alps_photo_depth_query_resolution_depth_scaled_1000/extracted_query_resolution_r50delg_gldv2clean --divide 1.0"

echo $cmd

LD_LIBRARY_PATH="/homes/kazi/ibrejcha/lib64:/usr/local/share/cuda-10.2/lib64/:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=$gpu $cmd 

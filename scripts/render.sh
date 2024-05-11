getenv=True

export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export CUDA_HOME_11_0="/is/software/nvidia/cuda-11.0"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME_11_0/lib64:$LD_LIBRARY_PATH

export HF_HOME="/is/cluster/yxiu/.cache"
export PYOPENGL_PLATFORM="egl"
export PYTORCH_KERNEL_CACHE_PATH="/is/cluster/yxiu/.cache/torch"

export EXP_DIR=$1
export SUBJECT_NAME=$(basename $1/masked/07_C.jpg | cut -d"." -f1)
export PYTHONPATH=$PYTHONPATH:$(pwd)

source /home/yxiu/miniconda3/bin/activate TeCH

export PROMPT=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f1)
export GENDER=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f2)


# [Optional] export textured mesh with UV map, using atlas for UV unwraping.
python cores/main.py --config configs/tech_texture_export.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME --test

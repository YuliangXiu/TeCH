getenv=True

export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export CUDA_HOME_11_0="/is/software/nvidia/cuda-11.0"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME_11_0/lib64:$LD_LIBRARY_PATH

export HF_HOME="/is/cluster/yxiu/.cache"
export PYOPENGL_PLATFORM="egl"

export INPUT_FILE=$1
export EXP_DIR=$2
export SUBJECT_NAME=$(basename $1/masked/07_C.jpg | cut -d"." -f1)
export PYTHONPATH=$PYTHONPATH:$(pwd)

source /home/yxiu/miniconda3/bin/activate TeCH

# Step 1: Preprocess image, get SMPL-X & normal estimation
python utils/body_utils/preprocess.py --in_path ${INPUT_FILE} --out_dir ${EXP_DIR} -nocrop

# Step 2: Get BLIP prompt and gender, you can also use your own prompt
python utils/get_prompt_blip.py --img-path ${EXP_DIR}/png/${SUBJECT_NAME}_crop.png --out-path ${EXP_DIR}/prompt.txt
export PROMPT=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f1)
export GENDER=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f2)

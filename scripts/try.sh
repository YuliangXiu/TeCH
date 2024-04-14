# set -x

getenv=True
source /home/yxiu/miniconda3/bin/activate TeCH


export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export CUDA_HOME_11_0="/is/software/nvidia/cuda-11.0"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME_11_0/lib64:$LD_LIBRARY_PATH
export HF_HOME="/is/cluster/yxiu/.cache"
export PYOPENGL_PLATFORM="egl"

export INPUT_FILE=$1
export EXP_DIR=$2
export SUBJECT_NAME=$(basename $1/masked/07_C.jpg | cut -d"." -f1)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 1: Preprocess image, get SMPL-X & normal estimation
python utils/body_utils/preprocess.py --in_path ${INPUT_FILE} --out_dir ${EXP_DIR} -nocrop

# # Step 2: Get BLIP prompt and gender, you can also use your own prompt
# python utils/get_prompt_blip.py --img-path ${EXP_DIR}/png/${SUBJECT_NAME}_crop.png --out-path ${EXP_DIR}/prompt.txt
# export PROMPT=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f1)
# export GENDER=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f2)

# # Step 3: Finetune Dreambooth model (minimal GPU memory requirement: 2x32G)
# # rm -rf ${EXP_DIR}/ldm
# python utils/ldm_utils/main.py -t --data_root ${EXP_DIR}/png/ --logdir ${EXP_DIR}/ldm/ \
#     --reg_data_root data/dreambooth_data/class_${GENDER}_images/ \
#     --bg_root data/dreambooth_data/bg_images/ \
#     --class_word ${GENDER} --no-test --gpus 0

# # Convert Dreambooth model to diffusers format
# python utils/ldm_utils/convert_ldm_to_diffusers.py \
#     --checkpoint_path ${EXP_DIR}/ldm/_v1-finetune_unfrozen/checkpoints/last.ckpt \
#     --original_config_file utils/ldm_utils/configs/stable-diffusion/v1-inference.yaml \
#     --scheduler_type ddim --image_size 512 --prediction_type epsilon --dump_path ${EXP_DIR}/sd_model


# # Step 4: Run geometry stage (Run on a single GPU)
# python cores/main.py --config configs/tech_geometry.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME
# python utils/body_utils/postprocess.py --dir $EXP_DIR/obj --name $SUBJECT_NAME

# # Step 5: Run texture stage (Run on a single GPU)
# python cores/main.py --config configs/tech_texture.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME

# # [Optional] export textured mesh with UV map, using atlas for UV unwraping.
# python cores/main.py --config configs/tech_texture_export.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME --test

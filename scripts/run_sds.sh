getenv=True

export CUDA_HOME="/is/software/nvidia/cuda-12.1"
export CUDA_HOME_11_0="/is/software/nvidia/cuda-11.0"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME_11_0/lib64:$LD_LIBRARY_PATH

export HF_HOME="/is/cluster/yxiu/.cache"
export PYOPENGL_PLATFORM="egl"
export PYTORCH_KERNEL_CACHE_PATH="/is/cluster/yxiu/.cache/torch"

export INPUT_FILE=$1
export EXP_DIR=$2
export SUBJECT_NAME=$(basename $1/masked/07_C.jpg | cut -d"." -f1)
export PYTHONPATH=$PYTHONPATH:$(pwd)

source /home/yxiu/miniconda3/bin/activate TeCH

export PROMPT=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f1)
export GENDER=$(cat ${EXP_DIR}/prompt.txt | cut -d'|' -f2)

rm -rf $EXP_DIR/texture/checkpoints
rm -rf $EXP_DIR/geometry/checkpoints

rm -rf $EXP_DIR/obj/07_C_pose.obj
rm -rf $EXP_DIR/obj/07_C_apose.obj
rm -rf $EXP_DIR/obj/07_C_smpl_da.obj
rm -rf $EXP_DIR/obj/07_C_geometry.obj
rm -rf $EXP_DIR/obj/07_C_tech*
rm -rf $EXP_DIR/obj/07_C_texture.obj
rm -rf $EXP_DIR/obj/07_C_texture_albedo.png

if ! [ -f $EXP_DIR/obj/07_C_pose.obj ]; then
    # Step 4: Run geometry stage (Run on a single GPU)
    python cores/main.py --config configs/tech_geometry.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME
    python utils/body_utils/postprocess.py --dir $EXP_DIR/obj --name $SUBJECT_NAME --gender $GENDER
fi

if ! [ -f $EXP_DIR/obj/07_C_texture.obj ]; then
    # Step 5: Run texture stage (Run on a single GPU)
    python cores/main.py --config configs/tech_texture.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME
fi

if ! [ -f $EXP_DIR/obj/07_C_texture_albedo.png ]; then
    # [Optional] export textured mesh with UV map, using atlas for UV unwraping.
    python cores/main.py --config configs/tech_texture_export.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME --test
fi

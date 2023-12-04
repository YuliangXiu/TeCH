getenv=True
source /home/yxiu/miniconda3/bin/activate TeCH   # user-defined
export HF_HOME="/is/cluster/yxiu/.cache"         # user-defined
export CUDA_HOME="/is/software/nvidia/cuda-11.7" # user-defined
export PYOPENGL_PLATFORM="egl"


python multi_concepts/train.py \
  --instance_data_dir examples/multi_concepts/creature  \
  --num_of_assets 3 \
  --initializer_tokens creature bowl stone \
  --class_data_dir data/multi_concepts_data \
  --phase1_train_steps 400 \
  --phase2_train_steps 400 \
  --output_dir results/multi_concepts/creature
getenv=True
source /home/yxiu/miniconda3/bin/activate TeCH   # user-defined
export HF_HOME="/is/cluster/yxiu/.cache"         # user-defined
export CUDA_HOME="/is/software/nvidia/cuda-11.7" # user-defined
export PYOPENGL_PLATFORM="egl"

python multi_concepts/inference.py \
  --model_path "results/multi_concepts/creature" \
  --prompt "a photo of <asset0> at the beach" \
  --output_path "results/multi_concepts/creature/output/result1.jpg"

python multi_concepts/inference.py \
  --model_path "results/multi_concepts/creature" \
  --prompt "an oil painting of <asset0> and <asset2>" \
  --output_path "results/multi_concepts/creature/output/result2.jpg"
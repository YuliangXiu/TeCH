## Environment setup

1. Install system packages

```bash
# install libraries
apt-get install -y \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libosmesa6-dev \
```

2. Install conda, [PyTorch3D](https://pytorch.org/get-started/locally/), [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) and other required packages

```bash
conda create --name TeCH python=3.10
conda activate TeCH
conda install pytorch torchvision tobarchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python install_pytorch3d.py
IGNORE_TORCH_VER=1 pip install git+https://github.com/NVIDIAGameWorks/kaolin.git
```

3. Build modules

```bash
cd core/lib/freqencoder
python setup.py install

cd ../gridencoder
python setup.py install

cd ../../thirdparties/nvdiffrast
python setup.py install
```

4. Download necessary data for body models: `bash scripts/download_body_data.sh`
5. Download pretrained models of MODNet: `bash scripts/download_modnets.sh`
6. Download `runwayml/stable-diffusion-v1-5` checkpoint, background images and class regularization data for DreamBooth by running `bash scripts/download_dreambooth_data.sh`, you can also try using another version of SD model, or use other images of `man` and `woman` for regularization (We simply generates these data with the SD model).
   
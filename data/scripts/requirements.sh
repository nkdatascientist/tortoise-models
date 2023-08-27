# Tortoise 🚀 by Nishanth M, Apache 2.0 License                  

# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣻⣿⣿⣿⣁⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⣀⣤⣤⣤⣤⣠⡶⣿⡏⠉⠉⠉⠉⢙⣿⢷⣦⣤⣤⣤⣤⣀⡀⠀⠀⠀
# ⠀⢀⣴⣿⣿⣿⣿⣿⣿⡟⠀⠈⣻⣶⣶⣶⣶⣿⠃⠀⠙⣿⣿⣿⣿⣿⣿⣷⣄⠀
# ⠀⠈⠛⠛⠛⠛⠛⠋⣿⠀⠀⣼⠏⠀⠀⠀⠀⠘⢧⡀⠀⢻⡏⠛⠛⠛⠛⠛⠉⠀
# ⠀⠀⠀⠀⠀⠀⠀⠸⣿⠀⠈⠻⣦⡀⠀⠀⢀⣴⠟⠁⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⣿⡀⠀⠀⣸⡿⠒⠒⢿⣏⠀⠀⠀⣾⠃⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⡀⠰⣿⡀⠀⠀⢀⣽⠗⠀⣼⠏⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⢀⣤⣾⣿⣄⠈⠁⠀⠀⠈⠁⣠⣾⣿⣤⡀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⡿⠿⢶⣤⣤⡶⠾⠿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⣿⣿⡿⠟⠉⠀⠀⠀⠻⡿⠁⠀⠀⠈⠛⠿⣿⣿⡆⠀⠀⠀⠀

#!/bin/bash

export PIP_ROOT_USER_ACTION=ignore
export DEBIAN_FRONTEND=noninteractive

ENV_NAME=$1 # "github"
MINICONDA_ROOT_PATH=$2 # "/home/nk/anaconda3/envs/"
MINICONDA_ENV_PATH=$MINICONDA_ROOT_PATH$ENV_NAME
Python="$MINICONDA_ENV_PATH/bin/python3.8"
echo "Environment: " $ENV_NAME
echo "Calling: " $Python
echo ""; sleep 5

$Python -m pip install --upgrade pip

# AIMET Version: 
    # https://github.com/quic/aimet/releases/tag/1.27.0
    # https://quic.github.io/aimet-pages/releases/1.27.0/install/index.html

sudo apt-get update -y && apt-get upgrade -y
sudo apt-get install -y apt-utils \
                        libcublas-11-0 \
                        libcufft-11-0 \
                        libcurand-11-0 \
                        libcusolver-11-0 \
                        libcusparse-11-0 \
                        libcudnn8 \
                        libnccl-dev \
                        libnccl2 \
                        cuda-command-line-tools-11-0 \
                        liblapacke-dev

if [ $3 = "torch" ]; then
    # https://stackoverflow.com/a/1774043
    $Python -m pip install tqdm \
                            pyyaml  \
                            pycocotools \
                            loguru
    $Python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/AimetTorch-torch_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/AimetCommon-torch_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/Aimet-torch_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
else
    $Python -m pip install tensorboard==2.10.1 \
                            tensorboardX==2.6 \
                            keras==2.10.0 \
                            h5py==2.10.0 \
                            tensorflow-gpu==2.10.1 \
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/AimetTensorflow-tf_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/AimetCommon-tf_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/Aimet-tf_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
fi

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

# AIMET Version: 
    # https://github.com/quic/aimet/releases/tag/1.27.0
    # https://quic.github.io/aimet-pages/releases/1.27.0/install/index.html

# sudo apt-get update -y && apt-get upgrade -y
sudo apt-get install -y apt-utils \
                        libavcodec-dev \
                        libavformat-dev \
                        libeigen3-dev \
                        libgtest-dev \
                        libgtk2.0-dev \
                        libsox-dev \
                        libsox-fmt-all \
                        libstdc++6:i386 \
                        libswscale-dev \
                        libxtst6 \

$Python -m pip install pip==23.0.0
if [ $3 = "torch" ]; then
    # $Python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    $Python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    # https://stackoverflow.com/a/1774043
    $Python -m pip install tqdm \
                            pyyaml  \
                            pycocotools \
                            loguru \
                            torchinfo \
                            ptflops \
                            scikit-image \
                            munch
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.26.1/Aimet-torch_gpu_1.26.1-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.26.1/AimetTorch-torch_gpu_1.26.1-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.26.1/AimetCommon-torch_gpu_1.26.1-cp38-cp38-linux_x86_64.whl
    sudo cat $MINICONDA_ROOT_PATH$ENV_NAME/lib/python3.8/site-packages/aimet_common/bin/reqs_deb_common.txt | xargs sudo apt-get --assume-yes install
    # sudo cat $MINICONDA_ROOT_PATH$ENV_NAME/lib/python3.8/site-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs sudo apt-get --assume-yes install
else
    $Python -m pip install tensorboard==2.10.1 \
                            tensorboardX==2.6 \
                            keras==2.10.0 \
                            h5py==2.10.0 \
                            tensorflow-gpu==2.10.1 
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/AimetTensorflow-tf_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/AimetCommon-tf_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
    $Python -m pip install https://github.com/quic/aimet/releases/download/1.27.0/Aimet-tf_gpu_1.27.0-cp38-cp38-linux_x86_64.whl
fi

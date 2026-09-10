# Install Guide

## 1. Prerequisites

Recommended:

- Linux
- RTX4090
- Python `3.8`


## 2. Create A Conda Environment

```bash
conda create -n artgym python=3.8 -y
conda activate artgym
```
## 3. Install Pytorch

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```
if you are using RTX 50s GPU, you need to install pytorch+cu12.x compatible with isaacgym.

## 3. Install Isaac Gym

This repo uses IsaacGym_TacSL, install it as follows:

```bash
pip install gdown && \
gdown 1nhLF4cKeUokCqU5LvEu8uYdEQtyJolXl && \
tar zxvf IsaacGym_Preview_TacSL_Package.tar.gz && \
pip install -e ./IsaacGym_Preview_TacSL_Package/isaacgym/python
```

## 4. Install Python Packages Used By This Repo

Install the core runtime packages:

```bash
pip install \
  gymnasium \
  numpy \
  scipy \
  matplotlib \
  imageio \
  hydra-core \
  omegaconf \
  gym \
  tensorboard \
  tensorboardX \
  pyyaml \
  psutil \
  setproctitle \
  opencv-python \
  wandb \
  pyvirtualdisplay \
  imageio[ffmpeg]
```

## 5. Install The Local `rl_games`

This workspace includes a local fork:

```text
rl_games/
```

Install it in editable mode:

```bash
pip install -e ./rl_games
```

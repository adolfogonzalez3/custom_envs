#! /bin/bash
# Script to install dependencies for linux systems using miniconda.
# Assumes no prior installation of miniconda and the OS is linux

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/condabin/conda create --channel conda-forge -yn tensorflow tensorflow numpy scipy pillow numexpr matplotlib mpi4py tqdm pyzmq opencv git-lfs

~/miniconda3/condabin/conda run -n tensorflow pip install gym

git clone -q https://github.com/hill-a/stable-baselines.git
~/miniconda3/condabin/conda run -n tensorflow pip install -qe stable-baselines/

git clone -q https://github.com/adolfogonzalez3/custom_envs.git
cd custom_envs && git checkout Working-on-Optimize
cd ~
~/miniconda3/condabin/conda run -n tensorflow pip install -qe custom_envs/


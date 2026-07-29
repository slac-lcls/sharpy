#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --constraint=OS_VER:9.6
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=sharpy_benchmark_strong
#SBATCH --output=/sdf/home/d/dnyanhet/sharpy_fresh/job_logs/benchmark_strong_%j.out
# rank count set at submit time: sbatch --nodes=N --ntasks=N run_benchmark_strong.sh

export OPAL_PREFIX=/sdf/sw/openmpi/v4.1.6/ompi/build

NV=/sdf/home/d/dnyanhet/.conda/envs/sharpy/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$NV/cublas/lib:$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib

export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:$NVIDIA_LIBS:$LD_LIBRARY_PATH

cd /sdf/home/d/dnyanhet/sharpy_fresh/sharpy

srun --mpi=pmix /sdf/home/d/dnyanhet/.conda/envs/sharpy/bin/python \
    sharpy_mpi_skeleton.py --benchmark-strong

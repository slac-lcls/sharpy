#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --constraint=OS_VER:9.6
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=sharpy_object_halo_4rank
#SBATCH --output=/sdf/home/d/dnyanhet/sharpy_fresh/job_logs/object_halo_4rank_%j.out

# mpi4py in the sharpy conda env was built against this OpenMPI (matches the
# ampere nodes) -- both need to be set before python/mpi4py can find libmpi.
export OPAL_PREFIX=/sdf/sw/openmpi/v4.1.6/ompi/build

# cupy-cuda12x is a pip-only wheel with no bundled/system CUDA toolkit on this
# cluster -- its runtime libs (libcublas.so.12 etc.) live inside the separately
# pip-installed nvidia-*-cu12 packages and aren't found automatically.
NV=/sdf/home/d/dnyanhet/.conda/envs/sharpy/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$NV/cublas/lib:$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib

export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:$NVIDIA_LIBS:$LD_LIBRARY_PATH

cd /sdf/home/d/dnyanhet/sharpy_fresh/sharpy

srun --mpi=pmix /sdf/home/d/dnyanhet/.conda/envs/sharpy/bin/python \
    validate_object_halo.py

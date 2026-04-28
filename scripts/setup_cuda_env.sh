#!/bin/bash
# Source this file to make CuPy and other CUDA-12 libraries findable.
#
# The torch installation already bundles all needed CUDA 12 runtime libs
# (libcublas, libnvrtc, libcusparse, libcusolver, ...) under
# ~/.local/lib/python3.9/site-packages/nvidia/*/lib
# but their directories are not on LD_LIBRARY_PATH by default.
#
# Usage:
#   source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh
# or prepend in SLURM worker scripts.

_NVIDIA_ROOT="/home/leo07010/.local/lib/python3.9/site-packages/nvidia"
if [ -d "$_NVIDIA_ROOT" ]; then
    _CUDA_LIBS=""
    for sub in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft cufile \
               curand cusolver cusparse cusparselt nccl nvjitlink nvtx; do
        d="$_NVIDIA_ROOT/$sub/lib"
        if [ -d "$d" ]; then
            _CUDA_LIBS="${_CUDA_LIBS}${d}:"
        fi
    done
    export LD_LIBRARY_PATH="${_CUDA_LIBS}${LD_LIBRARY_PATH:-}"
fi
unset _NVIDIA_ROOT _CUDA_LIBS sub d

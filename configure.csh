#!/bin/csh

set path = ( /usr/workspace/wsb/brain/utils/toss2/cmake-3.9.6/bin  $path )
setenv CMAKE_PREFIX_PATH /usr/global/tools/mpi/sideinstalls/$SYS_TYPE/mvapich2-2.3/install-gcc-4.9.3-cuda-9.1

cmake -DCMAKE_INSTALL_PREFIX=../install \
      -DNCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.2.12-1+cuda9.0_x86_64 \
      -DALUMINUM_ENABLE_CUDA=ON \
      -DALUMINUM_ENABLE_NCCL=ON \
      -DALUMINUM_ENABLE_MPI_CUDA=OFF \
      ..

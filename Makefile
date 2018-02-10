cur_dir = $(shell pwd)
CXXFLAGS += -Wall -Wextra -pedantic -Wshadow -O3 -std=c++11 -fopenmp -g -fPIC -lhwloc -I$(cur_dir)/src -I$(cur_dir)/test
LIB = -L$(cur_dir) -lallreduce -Wl,-rpath=$(cur_dir) -lrt
NVCCFLAGS += -arch sm_30 -I$(cur_dir)/src -I$(cur_dir)/test -std=c++11

# NCCL2 is available at:
# - ray: /usr/workspace/wsb/brain/nccl2/nccl-2.0.5-3+cuda8.0_ppc64el
# - surface: /usr/workspace/wsb/brain/nccl2/nccl-2.0.5+cuda8.0
NCCL_DIR =
ifeq ($(ENABLE_NCCL), YES)
	ENABLE_CUDA = YES
	CXXFLAGS += -I$(NCCL_DIR)/include  -DALUMINUM_HAS_NCCL
	LIB += -L$(NCCL_DIR)/lib -lnccl -Wl,-rpath=$(NCCL_DIR)/lib
endif

ifeq ($(ENABLE_MPI_CUDA), YES)
	ENABLE_CUDA = YES
	CXXFLAGS += -DALUMINUM_HAS_MPI_CUDA -DALUMINUM_MPI_CUDA_DEBUG
	CUDA_OBJ = src/mpi_cuda/cuda_kernels.o
	MPI_CUDA_HEADERS = src/allreduce_mpi_cuda_impl.hpp test/test_utils_mpi_cuda.hpp src/mpi_cuda/allreduce.hpp src/mpi_cuda/allreduce_ring.hpp
endif

ifeq ($(ENABLE_CUDA), YES)
	CUDA_HOME = $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(shell which nvcc)))))
	CXXFLAGS += -I$(CUDA_HOME)/include -DALUMINUM_HAS_CUDA
	LIB += -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath=$(CUDA_HOME)/lib64
endif

all: liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_overlap benchmark_reductions test_correctness test_multi_nballreduces

liballreduce.so: src/allreduce.cpp src/allreduce_mpi_impl.cpp src/allreduce.hpp src/allreduce_impl.hpp src/allreduce_mempool.hpp src/allreduce_mpi_impl.hpp src/tuning_params.hpp src/allreduce_nccl_impl.hpp $(CUDA_OBJ) $(MPI_CUDA_HEADERS)
	mpicxx $(CXXFLAGS) -shared -o liballreduce.so src/allreduce.cpp src/allreduce_mpi_impl.cpp

benchmark_allreduces: liballreduce.so benchmark/benchmark_allreduces.cpp src/allreduce_nccl_impl.hpp $(CUDA_OBJ) $(MPI_CUDA_HEADERS)
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_allreduces benchmark/benchmark_allreduces.cpp $(CUDA_OBJ) 

benchmark_nballreduces: liballreduce.so benchmark/benchmark_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_nballreduces benchmark/benchmark_nballreduces.cpp

benchmark_overlap: liballreduce.so benchmark/benchmark_overlap.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_overlap benchmark/benchmark_overlap.cpp

test_correctness: liballreduce.so test/test_correctness.cpp src/allreduce_nccl_impl.hpp test/test_utils.hpp $(CUDA_OBJ) $(MPI_CUDA_HEADERS)
	mpicxx $(CXXFLAGS) $(LIB) -o test_correctness test/test_correctness.cpp $(CUDA_OBJ)

test_multi_nballreduces: liballreduce.so test/test_multi_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_multi_nballreduces test/test_multi_nballreduces.cpp

benchmark_reductions: benchmark/benchmark_reductions.cpp
	mpicxx $(CXXFLAGS) -o benchmark_reductions benchmark/benchmark_reductions.cpp

src/mpi_cuda/cuda_kernels.o: src/mpi_cuda/cuda_kernels.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_reductions test_correctness test_multi_nballreduces benchmark_overlap

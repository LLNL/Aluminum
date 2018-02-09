CXXFLAGS += -Wall -Wextra -pedantic -Wshadow -O3 -std=c++11 -fopenmp -g -fPIC -lhwloc 
cur_dir = $(shell pwd)
LIB = -L$(cur_dir) -lallreduce -Wl,-rpath=$(cur_dir) -lrt

# NCCL2 is available at:
# - ray: /usr/workspace/wsb/brain/nccl2/nccl-2.0.5-3+cuda8.0_ppc64el
# - surface: /usr/workspace/wsb/brain/nccl2/nccl-2.0.5+cuda8.0
NCCL_DIR = 
ifeq ($(ENABLE_NCCL), YES)
	ENABLE_CUDA = YES
	CXXFLAGS += -I$(NCCL_DIR)/include  -DALUMINUM_HAS_NCCL
	LIB += -L$(NCCL_DIR)/lib -lnccl -Wl,-rpath=$(NCCL_DIR)/lib
endif

ifeq ($(ENABLE_CUDA), YES)
	CUDA_HOME = $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(shell which nvcc)))))
	CXXFLAGS += -I$(CUDA_HOME)/include -DALUMINUM_HAS_CUDA
	LIB += -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath=$(CUDA_HOME)/lib64
endif

all: liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_overlap benchmark_reductions test_correctness test_multi_nballreduces

liballreduce.so: allreduce.cpp allreduce_mpi_impl.cpp allreduce.hpp allreduce_impl.hpp allreduce_mempool.hpp allreduce_mpi_impl.hpp tuning_params.hpp allreduce_nccl_impl.hpp
	mpicxx $(CXXFLAGS) -shared -o liballreduce.so allreduce.cpp allreduce_mpi_impl.cpp

benchmark_allreduces: liballreduce.so benchmark_allreduces.cpp allreduce_nccl_impl.hpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_allreduces benchmark_allreduces.cpp

benchmark_nballreduces: liballreduce.so benchmark_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_nballreduces benchmark_nballreduces.cpp

benchmark_overlap: liballreduce.so benchmark_overlap.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_overlap benchmark_overlap.cpp

test_correctness: liballreduce.so test_correctness.cpp allreduce_nccl_impl.hpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_correctness test_correctness.cpp

test_multi_nballreduces: liballreduce.so test_multi_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_multi_nballreduces test_multi_nballreduces.cpp

benchmark_reductions: benchmark_reductions.cpp
	mpicxx $(CXXFLAGS) -o benchmark_reductions benchmark_reductions.cpp

clean:
	rm -f liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_reductions test_correctness test_multi_nballreduces benchmark_overlap

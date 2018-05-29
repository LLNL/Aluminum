cur_dir = $(shell pwd)
MPICXX ?= mpicxx
NVCC ?= nvcc
SED ?= $(shell which sed)

CXXFLAGS += -Wall -Wextra -pedantic -Wshadow -O3 -std=gnu++11 -fopenmp -g -fPIC -lhwloc -I$(cur_dir)/src -I$(cur_dir)/test
LIB = -L$(cur_dir) -lAl -Wl,-rpath=$(cur_dir) -lrt
LIB_LIB = 
NVCCFLAGS += --compiler-bindir $(CXX) -arch sm_30 -I$(cur_dir)/src -I$(cur_dir)/test -std=c++11

ALUMINUM_DEFINES = AL_HAS_CUDA AL_HAS_MPI_CUDA AL_HAS_NCCL

# NCCL2 is available at:
# NOTE: The current NCCL 2 we have is based on cuda 8.0
# - ray: /usr/workspace/wsb/brain/nccl2/nccl_2.0.5-3+cuda8.0_ppc64el
# - surface: /usr/workspace/wsb/brain/nccl2/nccl-2.0.5+cuda8.0
ifeq ($(ENABLE_NCCL_CUDA), YES)
	ENABLE_CUDA = YES
	ifeq ($(shell hostname|grep surface -c), 1)
		NCCL_DIR = /usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64
	else ifeq ($(shell hostname|grep pascal -c), 1)
		NCCL_DIR = /usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64
	else ifeq ($(shell hostname|grep ray -c), 1)
	  	NCCL_DIR = /usr/workspace/wsb/brain/nccl2/nccl_2.0.5-3+cuda8.0_ppc64el
	endif
	CXXFLAGS += -I$(NCCL_DIR)/include
	LIB += -L$(NCCL_DIR)/lib -lnccl -Wl,-rpath=$(NCCL_DIR)/lib
	LIB_LIB += -L$(NCCL_DIR)/lib -lnccl -Wl,-rpath=$(NCCL_DIR)/lib

	AL_HAS_NCCL=YES
endif

ifeq ($(ENABLE_MPI_CUDA), YES)
	ENABLE_CUDA = YES
	CUDA_OBJ = src/mpi_cuda/cuda_kernels.o
	MPI_CUDA_HEADERS = src/mpi_cuda_impl.hpp test/test_utils_mpi_cuda.hpp src/mpi_cuda/allreduce.hpp src/mpi_cuda/allreduce_ring.hpp

	AL_HAS_MPI_CUDA=YES
endif


ifeq ($(ENABLE_CUDA), YES)
	CUDA_HOME = $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(shell which $(NVCC))))))
	CXXFLAGS += -I$(CUDA_HOME)/include
	LIB += -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath=$(CUDA_HOME)/lib64
	LIB_LIB += -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath=$(CUDA_HOME)/lib64

	AL_HAS_CUDA=YES
endif
export ${ALUMINUM_DEFINES}

all: libAl.so benchmark_allreduces benchmark_nballreduces benchmark_overlap benchmark_reductions test_correctness test_multi_nballreduces test_nccl_collectives

Al_config.hpp: cmake/Al_config.hpp.in
	@set -- && \
	for var in ${ALUMINUM_DEFINES}; do \
	    eval val=\$$$${var} && \
	    if [ "YES" = "$${val}" ]; then \
		set -- "$$@" -e "s|cmakedefine $${var}|define $${var}|"; \
	    fi; \
	done && \
	set -- "$$@" -e "s|#cmakedefine \(.*\)|/*#undef \1*/|g" && \
	sed "$$@" cmake/Al_config.hpp.in > src/Al_config.hpp

libAl.so: Al_config.hpp src/Al.cpp src/mpi_impl.cpp src/Al.hpp src/mempool.hpp src/mpi_impl.hpp src/tuning_params.hpp src/nccl_impl.hpp src/nccl_impl.cpp src/mpi_cuda_impl.cpp
	$(MPICXX) $(CXXFLAGS) $(LIB_LIB) -shared -o libAl.so src/Al.cpp src/mpi_impl.cpp src/nccl_impl.cpp src/mpi_cuda_impl.cpp

benchmark_allreduces: libAl.so benchmark/benchmark_allreduces.cpp  $(CUDA_OBJ) $(MPI_CUDA_HEADERS)
	$(MPICXX) $(CXXFLAGS) $(LIB) -o benchmark_allreduces benchmark/benchmark_allreduces.cpp $(CUDA_OBJ) 

benchmark_nballreduces: libAl.so benchmark/benchmark_nballreduces.cpp
	$(MPICXX) $(CXXFLAGS) $(LIB) -o benchmark_nballreduces benchmark/benchmark_nballreduces.cpp

benchmark_overlap: libAl.so benchmark/benchmark_overlap.cpp
	$(MPICXX) $(CXXFLAGS) $(LIB) -o benchmark_overlap benchmark/benchmark_overlap.cpp

test_correctness: libAl.so test/test_correctness.cpp src/nccl_impl.hpp test/test_utils.hpp $(CUDA_OBJ) $(MPI_CUDA_HEADERS)
	$(MPICXX) $(CXXFLAGS) $(LIB) -o test_correctness test/test_correctness.cpp $(CUDA_OBJ)

test_nccl_collectives: libAl.so test/test_nccl_collectives.cpp src/nccl_impl.hpp test/test_utils.hpp $(CUDA_OBJ) $(MPI_CUDA_HEADERS)
	$(MPICXX) $(CXXFLAGS) $(LIB) -o test_nccl_collectives test/test_nccl_collectives.cpp $(CUDA_OBJ)

test_multi_nballreduces: libAl.so test/test_multi_nballreduces.cpp
	$(MPICXX) $(CXXFLAGS) $(LIB) -o test_multi_nballreduces test/test_multi_nballreduces.cpp

benchmark_reductions: benchmark/benchmark_reductions.cpp
	$(MPICXX) $(CXXFLAGS) -o benchmark_reductions benchmark/benchmark_reductions.cpp

src/mpi_cuda/cuda_kernels.o: src/mpi_cuda/cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

clean:
	rm -f libAl.so benchmark_allreduces benchmark_nballreduces benchmark_reductions test_correctness test_multi_nballreduces test_nccl_collectives benchmark_overlap src/mpi_cuda/cuda_kernels.o src/Al_config.hpp

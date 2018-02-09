#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

#include <unistd.h>
#include <cstdlib>

#define COLL_ASSERT(x) do {                                      \
    int s = x;                                                   \
    if (!s) {                                                    \
      std::cerr << "Assertion failure at "                       \
                << __FILE__ << ":" << __LINE__ << std::endl;     \
      abort();                                                   \
    }                                                            \
  } while (0)

#ifdef COLL_DEBUG
#define COLL_LOG_DEBUG(x) x;
#else
#define COLL_LOG_DEBUG(x)
#endif


#define COLL_CHECK_CUDA(cuda_call)                                      \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define COLL_CHECK_MPI(call)                                            \
  do {                                                                  \
    int status = call;                                                  \
    if (status != MPI_SUCCESS) {                                        \
      std::cerr << "MPI error" << std::endl;                            \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, status);                                \
    }                                                                   \
  } while (0)


namespace allreduces {
namespace internal {
namespace mpi_cuda {

inline int dec(int x, int t) {
  return (x -1 + t) % t;
}

inline int inc(int x, int t) {
  return (x + 1) % t;
}

inline void *malloc_aligned(size_t s) {
#if 1
  unsigned long align_size = sysconf(_SC_PAGESIZE);
  //std::cerr << "page size: " << align_size << std::endl;
  void *p = nullptr;
  if (posix_memalign(&p, align_size, s)) {
    std::cerr << "posix_memalign failed\n";
    abort();
  }
  return p;  
#else
  return malloc(s);
#endif
}

inline
void create_streams(std::vector<cudaStream_t> &streams,
                    std::vector<int> &gpus) {
  COLL_ASSERT(streams.empty());
  int num_gpus = gpus.size();
  for (int i = 0; i < num_gpus; ++i) {
    COLL_CHECK_CUDA(cudaSetDevice(gpus[i]));
    cudaStream_t s;
    COLL_CHECK_CUDA(cudaStreamCreate(&s));
    streams.push_back(s);
  }
}

inline
void destroy_streams(std::vector<cudaStream_t> &streams,
                     std::vector<int> &gpus) {
  int num_gpus = gpus.size();
  for (int i = 0; i < num_gpus; ++i) {
    COLL_CHECK_CUDA(cudaSetDevice(gpus[i]));
    COLL_CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    COLL_CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
  streams.clear();
}

template <typename T>
MPI_Datatype get_mpi_data_type();

template <> inline
MPI_Datatype get_mpi_data_type<float>() {
  return MPI_FLOAT;
}

inline int get_mpi_comm_local_size() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (env == nullptr) {
    env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  }
  if (env == nullptr) {
    std::cerr << "Failed to determine the number of ranks per node" << std::endl;
    abort();
  }
  int size = atoi(env);
  return size;
}

class MPIPrintStream {
 public:
  MPIPrintStream(std::ostream &os, int rank): m_os(os) {
    ss << "[" << rank << "] ";
  }
  ~MPIPrintStream() {
    m_os << ss.str();
  }
  std::stringstream &operator()() {
    return ss;
  }
  
 protected:
  std::ostream &m_os;
  std::stringstream ss;
};

} // namespace mpi_cuda
} // namespace internal
} // namespace allreduces

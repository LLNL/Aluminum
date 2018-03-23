#pragma once

#include <memory>
#include <assert.h>
#include <cuda_runtime.h>
#include "test_utils.hpp"

#define NCCL_THRESHOLD	1e-05

#define CHECK_CUDA(cuda_call)                                           \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)

int get_number_of_gpus() {
  int num_gpus = 0;
  char *env = getenv("ALUMINUM_NUM_GPUS");
  if (env) {
    std::cout << "Number of GPUs set by ALUMINUM_NUM_GPUS\n";
    num_gpus = atoi(env);
  } else {
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));    
  }
  return num_gpus;
}

int get_local_rank() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("SLURM_LOCALID");
  if (!env) {
    std::cerr << "Can't determine local rank\n";
    abort();
  }
  return atoi(env);
}

int get_local_size() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = getenv("SLURM_NTASKS_PER_NODE");
  if (!env) {
    std::cerr << "Can't determine local size\n";
    abort();
  }
  return atoi(env);
}

inline int set_device() {
  int num_gpus = get_number_of_gpus();
  int local_rank = get_local_rank();
  int local_size = get_local_size();
  if (num_gpus < local_size) {
    std::cerr << "Number of available GPUs is smaller than the number of local MPI ranks\n";
    abort();
  }    
  int device = local_rank;
  CHECK_CUDA(cudaSetDevice(device));
  return device;
}

template <typename T>
class CUDAVector {
 public:
  CUDAVector(): m_count(0), m_ptr(nullptr) {}
    
  CUDAVector(size_t count): m_count(count), m_ptr(nullptr) {
    allocate();
  }

  CUDAVector(const std::vector<T> &host_vector):
      m_count(host_vector.size()), m_ptr(nullptr) {
    allocate();
    CHECK_CUDA(cudaMemcpy(m_ptr, host_vector.data(), get_bytes(), cudaMemcpyDefault));
  }

  CUDAVector(const CUDAVector &v): m_count(v.m_count), m_ptr(nullptr) {
    allocate();
    CHECK_CUDA(cudaMemcpy(m_ptr, v.data(), get_bytes(), cudaMemcpyDefault));
  }

  CUDAVector(CUDAVector &&v): CUDAVector() {
    swap(*this, v);
  }
  
  ~CUDAVector() {
    clear();
  }

  friend void swap(CUDAVector &x, CUDAVector &y) {
    using std::swap;
    swap(x.m_count, y.m_count);
    swap(x.m_ptr, y.m_ptr);
  }

  size_t size() const {
    return m_count;
  }

  size_t get_bytes() const {
    return m_count * sizeof(T);
  }

  void clear() {
    if (m_count > 0) {
      CHECK_CUDA(cudaFree(m_ptr));
      m_ptr = nullptr;
      m_count = 0;
    }
  }

  void allocate() {
    assert(m_ptr == nullptr);
    if (m_count > 0) {
#if 0
      CHECK_CUDA(cudaMalloc(&m_ptr, get_bytes()));
#else
      cudaError_t e = cudaMalloc(&m_ptr, get_bytes());
      if (e != cudaSuccess) {
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "Error allocating "
                  << get_bytes() << " bytes of memory: "
                  << cudaGetErrorString(e)
                  << ", free: " << free_mem << "\n";
        cudaDeviceReset();
        abort();
      }
#endif
    }
  }

  CUDAVector &operator=(const CUDAVector<T> &v) {
    clear();
    m_count = v.m_count;
    allocate();
    CHECK_CUDA(cudaMemcpy(m_ptr, v.m_ptr, get_bytes(), cudaMemcpyDefault));
    return *this;
  }

  CUDAVector& move(const CUDAVector<T> &v) {
    CHECK_CUDA(cudaMemcpy(m_ptr, v.m_ptr, v.get_bytes(), cudaMemcpyDefault));
    return *this;
  }

  T *data() {
    return m_ptr;
  }

  const T *data() const {
    return m_ptr;
  }

  std::vector<T> copyout() const {
    std::vector<T> hv(size());
    CHECK_CUDA(cudaMemcpy(hv.data(), m_ptr, get_bytes(), cudaMemcpyDeviceToHost));
    return hv;
  }
  
  void copyin(const T *hp) {
    CHECK_CUDA(cudaMemcpy(m_ptr, hp, get_bytes(), cudaMemcpyHostToDevice));
  }

  void copyin(const std::vector<T> &hv) {
    clear();
    m_count = hv.size();
    allocate();
    copyin(hv.data());
  }
  
 private:
  size_t m_count;  
  T *m_ptr;
};


bool check_vector(const CUDAVector<float>& expected,
                  const CUDAVector<float>& actual) {
  std::vector<float> &&expected_host = expected.copyout();
  std::vector<float> &&actual_host= actual.copyout();
  return check_vector(expected_host, actual_host);
}

void get_expected_result(CUDAVector<float>& expected) {



  std::vector<float> &&host_data = expected.copyout();
  MPI_Allreduce(MPI_IN_PLACE, host_data.data(), expected.size(),
                MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  expected.copyin(host_data);
}

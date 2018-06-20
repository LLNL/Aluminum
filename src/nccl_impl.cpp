#include "nccl_impl.hpp"

// Error checking macros
#define CUDACHECK(cmd) do {                     \
  cudaError_t e = cmd;                          \
  if (e != cudaSuccess) {                       \
    printf("CUDA failure %s:%d '%s'\n",         \
           __FILE__, __LINE__,                  \
           cudaGetErrorString(e));              \
    exit(EXIT_FAILURE);                         \
  }                                             \
} while(0)
#define NCCLCHECK(cmd) do {                     \
  ncclResult_t r = cmd;                         \
  if (r!= ncclSuccess) {                        \
    printf("NCCL failure %s:%d '%s'\n",         \
           __FILE__, __LINE__,                  \
           ncclGetErrorString(r));              \
    exit(EXIT_FAILURE);                         \
  }                                             \
} while(0)

namespace Al {

// Initialize this.
const NCCLBackend::req_type NCCLBackend::null_req = (NCCLBackend::req_type) (-1);

NCCLCommunicator::NCCLCommunicator(MPI_Comm comm_)
  : MPICommunicator(comm_) {
  gpu_setup();
  nccl_setup();
}

NCCLCommunicator::~NCCLCommunicator() {
  nccl_destroy();
  CUDACHECK(cudaStreamDestroy(m_default_stream));
}

void NCCLCommunicator::gpu_setup() {
  CUDACHECK(cudaStreamCreate(&m_default_stream));
}

void NCCLCommunicator::nccl_setup() {
  // Get a unique ID for this communicator from NCCL and distribute it.
  ncclUniqueId nccl_id;
  if (rank() == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, get_comm());

  // This uses the current CUDA device.
  NCCLCHECK(ncclCommInitRank(&m_nccl_comm, size(), nccl_id, rank()));
}

void NCCLCommunicator::nccl_destroy() {
  NCCLCHECK(ncclCommDestroy(m_nccl_comm));
}

void NCCLCommunicator::synchronize() {
  CUDACHECK(cudaStreamSynchronize(m_default_stream));
}

cudaStream_t NCCLCommunicator::get_default_stream() {
  return m_default_stream;
}

void NCCLCommunicator::Allreduce(const void* sendbuf, void* recvbuf,
                                 size_t count, ncclDataType_t nccl_type,
                                 ncclRedOp_t nccl_redop,
                                 cudaStream_t default_stream) {
  if (count == 0) return;
  NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, nccl_type, nccl_redop,
                          m_nccl_comm, default_stream));
}

void NCCLCommunicator::Reduce(const void* sendbuf, void* recvbuf, size_t count,
                              ncclDataType_t nccl_type, ncclRedOp_t nccl_redop,
                              int root, cudaStream_t default_stream) {
  if (count == 0) return;
  NCCLCHECK(ncclReduce(sendbuf, recvbuf, count, nccl_type, nccl_redop, root,
                       m_nccl_comm, default_stream));
}

void NCCLCommunicator::Bcast(void* buf, size_t count,
                             ncclDataType_t nccl_type, int root,
                             cudaStream_t default_stream) {
  if (count == 0) return;
  NCCLCHECK(ncclBcast(buf, count, nccl_type, root, m_nccl_comm,
                      default_stream));
}

void NCCLCommunicator::Allgather(const void* sendbuf, void* recvbuf,
                                 size_t send_count, ncclDataType_t nccl_type,
                                 cudaStream_t default_stream) {
  if (send_count == 0) return;
  NCCLCHECK(ncclAllGather(sendbuf, recvbuf, send_count, nccl_type, m_nccl_comm, default_stream));
}

void NCCLCommunicator::Reduce_scatter(const void* sendbuf, void* recvbuf,
                                      size_t recv_count,
                                      ncclDataType_t nccl_type,
                                      ncclRedOp_t nccl_redop,
                                      cudaStream_t default_stream) {
  if (recv_count == 0) return;
  NCCLCHECK(ncclReduceScatter(sendbuf, recvbuf, recv_count, nccl_type,
                              nccl_redop, m_nccl_comm, default_stream));
}

}  // namespace Al

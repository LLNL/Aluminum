#include "nccl_impl.hpp"

namespace Al {

// Initialize this.
cudaEvent_t NCCLBackend::sync_event = (cudaEvent_t) 0;

NCCLCommunicator::NCCLCommunicator(MPI_Comm comm_, cudaStream_t stream_)
  : CUDACommunicator(comm_, stream_) {
  nccl_setup();
}

NCCLCommunicator::~NCCLCommunicator() {
  nccl_destroy();
}

void NCCLCommunicator::nccl_setup() {
  // Get a unique ID for this communicator from NCCL and distribute it.
  ncclUniqueId nccl_id;
  if (rank() == 0) {
    AL_CHECK_NCCL(ncclGetUniqueId(&nccl_id));
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, get_comm());

  // This uses the current CUDA device.
  AL_CHECK_NCCL(ncclCommInitRank(&m_nccl_comm, size(), nccl_id, rank()));
}

void NCCLCommunicator::nccl_destroy() {
  AL_CHECK_NCCL(ncclCommDestroy(m_nccl_comm));
}

namespace internal {
namespace nccl {

void init(int&, char**&) {
  AL_CHECK_CUDA(cudaEventCreateWithFlags(&NCCLBackend::sync_event,
                                         cudaEventDisableTiming));
}

void finalize() {
  AL_CHECK_CUDA(cudaEventDestroy(NCCLBackend::sync_event));
}

}  // namespace nccl
}  // namespace internal
}  // namespace Al

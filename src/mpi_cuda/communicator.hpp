#pragma once

#include "cudacommunicator.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

class RingMPICUDA;

class MPICUDACommunicator: public CUDACommunicator {
 public:
  MPICUDACommunicator() : MPICUDACommunicator(MPI_COMM_WORLD, 0) {}
  MPICUDACommunicator(cudaStream_t stream_) :
    MPICUDACommunicator(MPI_COMM_WORLD, stream_) {}
  MPICUDACommunicator(MPI_Comm comm_, cudaStream_t stream_);

  RingMPICUDA &get_ring() {
    return *m_ring;
  }

 protected:
  RingMPICUDA *m_ring;
};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al

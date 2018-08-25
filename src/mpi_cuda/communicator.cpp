#include "Al.hpp"
#include "mpi_cuda/communicator.hpp"
#include "mpi_cuda/allreduce_ring.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

MPICUDACommunicator::MPICUDACommunicator(MPI_Comm comm_, cudaStream_t stream_) :
    CUDACommunicator(comm_, stream_), m_ring(new RingMPICUDA(comm_, *this)) {
}

} // namespace mpi_cuda
} // namespace internal
} // namespace Al

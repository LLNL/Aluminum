#ifdef AL_HAS_MPI_CUDA
#include "mpi_cuda_impl.hpp"

namespace Al {

// Initialize this.
const MPICUDABackend::req_type MPICUDABackend::null_req = (MPICUDABackend::req_type) (-1);

}  // namespace Al

#endif  // AL_HAS_MPI_CUDA

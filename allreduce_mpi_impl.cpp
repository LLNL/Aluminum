#include "allreduce.hpp"

namespace allreduces {
namespace internal {
namespace mpi {

void init(int& argc, char**& argv) {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
      throw allreduce_exception("MPI_THREAD_MULTIPLE not provided");
    }
  }
}

void finalize() {
  int flag;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Finalize();
  }
}

}  // namespace mpi
}  // namespace internal
}  // namespace allreduces

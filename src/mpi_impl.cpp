#include "Al.hpp"

namespace Al {
namespace internal {
namespace mpi {

namespace {
// Whether we initialized MPI, or it was already initialized.
bool initialized_mpi = false;
}

void init(int& argc, char**& argv) {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
      throw_al_exception("MPI_THREAD_MULTIPLE not provided");
    }
    initialized_mpi = true;
  }
}

void finalize() {
  int flag;
  MPI_Finalized(&flag);
  if (!flag && initialized_mpi) {
    MPI_Finalize();
  }
}

}  // namespace mpi
}  // namespace internal
}  // namespace Al

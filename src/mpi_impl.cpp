////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "Al.hpp"

namespace Al {
namespace internal {
namespace mpi {

namespace {
// Whether we initialized MPI, or it was already initialized.
bool initialized_mpi = false;
// Maximum tag value in MPI.
int max_tag = 0;
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
  } else {
    // Ensure that we have THREAD_MULTIPLE.
    int provided;
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE) {
      throw_al_exception("MPI already initialized without MPI_THREAD_MULTIPLE");
    }
  }
  // Get the upper bound for tags; this is always set in MPI_COMM_WORLD.
  int* p;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &p, &flag);
  max_tag = *p;
}

void finalize() {
  int flag;
  MPI_Finalized(&flag);
  if (!flag && initialized_mpi) {
    MPI_Finalize();
  }
}

int get_max_tag() { return max_tag; }

}  // namespace mpi
}  // namespace internal
}  // namespace Al

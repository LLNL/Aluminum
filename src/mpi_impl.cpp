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

#include "aluminum/mpi_impl.hpp"
#include <mpi.h>
#include "aluminum/base.hpp"
#include "aluminum/mpi/communicator.hpp"

namespace Al {
namespace internal {
namespace mpi {

#ifdef AL_HAS_HALF
// Reduction operators for half.
MPI_Op half_sum_op;
MPI_Op half_prod_op;
MPI_Op half_min_op;
MPI_Op half_max_op;
#endif

namespace {
// Whether we initialized MPI, or it was already initialized.
bool initialized_mpi = false;
// Maximum tag value in MPI.
int max_tag = 0;

#ifdef AL_HAS_HALF
// Operator implementations for half.
void half_sum_reduction(void* invec_, void* inoutvec_, int* len,
                        MPI_Datatype*) {
  const __half* invec = reinterpret_cast<const __half*>(invec_);
  __half* inoutvec = reinterpret_cast<__half*>(inoutvec_);
  for (int i = 0; i < *len; ++i) {
    inoutvec[i] = __float2half(
      __half2float(invec[i]) + __half2float(inoutvec[i]));
  }
}

void half_prod_reduction(void* invec_, void* inoutvec_, int* len,
                        MPI_Datatype*) {
  const __half* invec = reinterpret_cast<const __half*>(invec_);
  __half* inoutvec = reinterpret_cast<__half*>(inoutvec_);
  for (int i = 0; i < *len; ++i) {
    inoutvec[i] = __float2half(
      __half2float(invec[i]) * __half2float(inoutvec[i]));
  }
}

void half_min_reduction(void* invec_, void* inoutvec_, int* len,
                        MPI_Datatype*) {
  const __half* invec = reinterpret_cast<const __half*>(invec_);
  __half* inoutvec = reinterpret_cast<__half*>(inoutvec_);
  for (int i = 0; i < *len; ++i) {
    inoutvec[i] = __float2half(
      std::min(__half2float(invec[i]), __half2float(inoutvec[i])));
  }
}

void half_max_reduction(void* invec_, void* inoutvec_, int* len,
                        MPI_Datatype*) {
  const __half* invec = reinterpret_cast<const __half*>(invec_);
  __half* inoutvec = reinterpret_cast<__half*>(inoutvec_);
  for (int i = 0; i < *len; ++i) {
    inoutvec[i] = __float2half(
      std::max(__half2float(invec[i]), __half2float(inoutvec[i])));
  }
}
#endif
}

void init(int& argc, char**& argv) {
  int flag;
#ifdef AL_MPI_SERIALIZE
  int required = MPI_THREAD_SERIALIZED;
#else
  int required = MPI_THREAD_MULTIPLE;
#endif
  MPI_Initialized(&flag);
  if (!flag) {
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
      throw_al_exception("Insufficient MPI thread support");
    }
    initialized_mpi = true;
  } else {
    // Ensure that we have sufficient threading support in MPI.
    int provided;
    MPI_Query_thread(&provided);
    if (provided < required) {
      throw_al_exception(
        "MPI already initialized with insufficient thread support");
    }
  }
  // Get the upper bound for tags; this is always set in MPI_COMM_WORLD.
  int* p;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &p, &flag);
  max_tag = *p;

#ifdef AL_HAS_HALF
  // Set up reduction operators for half.
  MPI_Op_create(&half_sum_reduction, true, &half_sum_op);
  MPI_Op_create(&half_prod_reduction, true, &half_prod_op);
  MPI_Op_create(&half_min_reduction, true, &half_min_op);
  MPI_Op_create(&half_max_reduction, true, &half_max_op);
#endif
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

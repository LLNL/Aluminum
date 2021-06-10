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

#pragma once

#include <mpi.h>

namespace Al {
namespace internal {

/** Provides a wrapper around an MPI_Comm and a compute stream.
 *
 * This will duplicate the provided MPI_Comm and will manage its
 * lifetime.
 *
 * The stream is assumed to be managed outside of the lifetime of the
 * communicator and will not be duplicated or deleted.
 */
template <typename Stream>
class MPICommAndStreamWrapper {
public:
  /** Wrap (a duplicate of) comm_. */
  MPICommAndStreamWrapper(MPI_Comm comm_, Stream stream_) :
    stream(stream_)
  {
    // Duplicate the communicator to avoid interference.
    MPI_Comm_dup(comm_, &comm);
    MPI_Comm_rank(comm, &rank_in_comm);
    MPI_Comm_size(comm, &size_of_comm);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &rank_in_local_comm);
    MPI_Comm_size(local_comm, &size_of_local_comm);
  }
  /** disable mpi */
  MPICommAndStreamWrapper(int rank_, int size_, Stream stream_) :
      stream(stream_), rank_in_comm(rank_), size_of_comm(size_), mpi_disabled(true) {
  }
  /** Cannot copy this. */
  MPICommAndStreamWrapper(const MPICommAndStreamWrapper& other) = delete;
  /** Default move constructor. */
  MPICommAndStreamWrapper(MPICommAndStreamWrapper&& other) = default;
  /** Cannot copy this. */
  MPICommAndStreamWrapper& operator=(const MPICommAndStreamWrapper& other) noexcept = delete;
  /** Default move assignment operator. */
  MPICommAndStreamWrapper& operator=(MPICommAndStreamWrapper&& other) = default;

  /** Destroy the underlying MPI_Comm. */
  ~MPICommAndStreamWrapper() {
    if (!mpi_disabled) {
      int finalized;
      MPI_Finalized(&finalized);
      if (!finalized) {
        MPI_Comm_free(&comm);
        MPI_Comm_free(&local_comm);
      }
    }
  }

  /** Create a new communicator with the same processes. */
  MPICommAndStreamWrapper copy() const { return MPICommAndStreamWrapper(comm); }

  /** Return the rank of the calling process in the communicator. */
  int rank() const { return rank_in_comm; }
  /** Return the number of processes in the communicator. */
  int size() const { return size_of_comm; }
  /**
   * Return the rank of the calling process among processes in the
   * same communicator on the same node.
   */
  int local_rank() const { return rank_in_local_comm; }
  /** Return the number of processes in the communicator on the same node. */
  int local_size() const { return size_of_local_comm; }

  /** Return the raw MPI communicator. */
  MPI_Comm get_comm() const { return comm; }
  /** Return the raw local MPI communicator. */
  MPI_Comm get_local_comm() const { return local_comm; }

  /** Return the assoicated compute stream. */
  Stream get_stream() const { return stream; }

private:
  /** Associated compute stream. */
  Stream stream;
  /** Associated MPI communicator. */
  MPI_Comm comm;
  /** Associated MPI communicator for the local node. */
  MPI_Comm local_comm;
  /** Rank in comm. */
  int rank_in_comm;
  /** Size of comm. */
  int size_of_comm;
  /** Rank in the local communicator. */
  int rank_in_local_comm;
  /** Size of the local communicator. */
  int size_of_local_comm;
  /** disable mpi. */
  bool mpi_disabled = false;
};

} // namespace internal
} // namespace Al

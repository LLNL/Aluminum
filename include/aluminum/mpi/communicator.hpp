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

namespace Al {
namespace internal {
namespace mpi {

int get_max_tag();

/**
 * Communicator for MPI-based collectives.
 */
class MPICommunicator : public Communicator {
 public:
  /** Default constructor; use MPI_COMM_WORLD. */
  MPICommunicator() : MPICommunicator(MPI_COMM_WORLD) {}
  /** Use a particular MPI communicator. */
  MPICommunicator(MPI_Comm comm_) : Communicator() {
    // Duplicate the communicator to avoid interference.
    MPI_Comm_dup(comm_, &comm);
    MPI_Comm_rank(comm, &rank_in_comm);
    MPI_Comm_size(comm, &size_of_comm);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &rank_in_local_comm);
    MPI_Comm_size(local_comm, &size_of_local_comm);
  }
  virtual ~MPICommunicator() override {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      // TODO: temporary skipping deleting comm
      //MPI_Comm_free(&comm);
      //MPI_Comm_free(&local_comm);
    }
  }
  Communicator* copy() const override { return new MPICommunicator(comm); }
  int rank() const override { return rank_in_comm; }
  int size() const override { return size_of_comm; }
  MPI_Comm get_comm() const { return comm; }
  int local_rank() const override { return rank_in_local_comm; }
  int local_size() const override { return size_of_local_comm; }
  MPI_Comm get_local_comm() const { return local_comm; }
  int get_free_tag() {
    int tag = free_tag++;
    if (free_tag >= internal::mpi::get_max_tag()
        || free_tag < starting_free_tag) {
      free_tag = starting_free_tag;
    }
    return tag;
  }

 private:
  /** Associated MPI communicator. */
  MPI_Comm comm;
  /** Rank in comm. */
  int rank_in_comm;
  /** Size of comm. */
  int size_of_comm;
  /** Communicator for the local node. */
  MPI_Comm local_comm;
  /** Rank in the local communicator. */
  int rank_in_local_comm;
  /** Size of the local communicator. */
  int size_of_local_comm;
  /**
   * Starting tag to use for non-blocking operations.
   * No other operations should use any tag >= to this one.
   */
  static constexpr int starting_free_tag = 10;
  /** Free tag for communication. */
  int free_tag = starting_free_tag;
};

} // namespace mpi
} // namespace internal
} // namespace Al

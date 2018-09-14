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
/**
 * Communicator including a CUDA stream to associate with operations.
 */
class CUDACommunicator : public MPICommunicator {
 public:
  CUDACommunicator(cudaStream_t stream_) :
    CUDACommunicator(MPI_COMM_WORLD, stream_) {}
  CUDACommunicator(MPI_Comm comm_, cudaStream_t stream_) :
    MPICommunicator(comm_), stream(stream_) {}
  Communicator* copy() const override {
    return new CUDACommunicator(get_comm(), stream);
  }
  /** Return the CUDA stream associated with this communicator. */
  cudaStream_t get_stream() const { return stream; }
  /** Set a new CUDA stream for this communicator. */
  void set_stream(cudaStream_t stream_) { stream = stream_; }
 private:
  /** CUDA stream associated with this communicator. */
  cudaStream_t stream;
};

}  // namespace Al

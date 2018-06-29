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

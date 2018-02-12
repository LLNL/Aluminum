#pragma once

#include "allreduce.hpp"
#include "nccl.h"
#include "common.h"

namespace allreduces {

enum class NCCLAllreduceAlgorithm {
  automatic
};

inline std::string allreduce_name(NCCLAllreduceAlgorithm algo) {
  switch (algo) {
  case NCCLAllreduceAlgorithm::automatic:
    return "automatic";
  default:
    return "unknown";
  }
}

/// We assume NCCL version 2.0 or higher for allreduce to work
class NCCLCommunicator : public Communicator {
//class NCCLCommunicator : public Communicator {
 public:
  /** Default constructor; use MPI_COMM_WORLD. */
  NCCLCommunicator() : NCCLCommunicator(MPI_COMM_WORLD) {}
  /** Use a particular MPI communicator. */
  NCCLCommunicator(MPI_Comm comm_) : Communicator() {

    /// Set up MPI communicator
    mpi_setup(comm_);

    /// Set up GPU-related informatiton
    gpu_setup();

    /// NCCL set up here
    nccl_setup();
  }

  ~NCCLCommunicator() override {
    // TODO: Fix; can't do this after finalization.
    //MPI_Comm_free(&comm);
  }

  Communicator* copy() const override { return new NCCLCommunicator(mpi_comm); }
  int rank() const override { return rank_in_comm; }
  int size() const override { return size_of_comm; }
  MPI_Comm get_comm() const { return mpi_comm; }
  int local_rank() const { return rank_in_local_comm; }
  int local_size() const { return size_of_local_comm; }
  MPI_Comm get_local_comm() const { return local_comm; }

protected:
  void mpi_setup(MPI_Comm comm_);

  void gpu_setup();

  void nccl_setup();

  void nccl_destroy();

  /// It is assumed that both sendbuf and recvbuf are in device memory
  /// for NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  void Allreduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
               ncclRedOp_t nccl_redop); 

 private:
  /** Associated MPI communicator. */
  MPI_Comm mpi_comm;
  /** Communicator for the local node. */
  MPI_Comm local_comm;

  /** Rank in comm. */
  int rank_in_comm;
  /** Size of comm. */
  int size_of_comm;

  /** Rank in the local communicator. */
  int rank_in_local_comm;
  /** Size of the local communicator. */
  int size_of_local_comm;

  /** List of GPU related variables. */
  /// List of GPUs to be used
  std::vector<int> m_gpus;
  /// List of CUDA streams
  std::vector<cudaStream_t> m_streams;
  /// Number of GPUs allocated to the current rank
  int m_num_gpus;
  /// Number of visible GPUs on this compute node
  int m_num_visible_gpus;

  /** List of NCCL 2 related variables. */
  /// NOTE: It is assumed that ONLY ONE GPU is allocated to one MPI rank
  std::vector<ncclComm_t> m_nccl_comm;
};
/*
class NCCLCommunicator : public Communicator {
 public:

  /// NCCL communicator MUST operate in conjunction with an MPI_Comm 
  /// Default constructor; use MPI_COMM_WORLD
  NCCLCommunicator() : NCCLCommunicator(MPI_COMM_WORLD) {}

  /// NCCLCommunicator with an MPI communicator given
  NCCLCommunicator(MPI_Comm comm_) : MPICommunicator(comm_) {

    mpicomm = get_comm();

        /// Set up GPU-related informatiton
    gpu_setup();

    /// NCCL set up here
    nccl_setup();
  }

  ~NCCLCommunicator() override {
    /// NCCL destroy here
    nccl_destroy();
  }


  /// It is assumed that both sendbuf and recvbuf are in device memory
  /// for NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  void Allreduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
                 ncclRedOp_t nccl_redop) {

    if(count == 0) return;
    int num_gpus_assigned = m_gpus.size();

    if(num_gpus_assigned > 1) ncclGroupStart();
    for(int i = 0; i < num_gpus_assigned; ++i) {
      CUDACHECK(cudaSetDevice(m_gpus[i]));
      NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, nccl_type, nccl_redop, m_nccl_comm[i], m_streams[i]));
    }
    if(num_gpus_assigned > 1) ncclGroupEnd();

  }

  void gpu_setup() {

    const int rank_in_node = local_rank();
    const int procs_per_node = local_size();

    /// Determine number of visible GPUs on the current node
    CUDACHECK(cudaGetDeviceCount(&m_num_visible_gpus));
    if(m_num_visible_gpus < 1) {
      std::cerr << "NCCLCommunicator: rank " << rank() << ": has no GPUs found on the node\n";
      MPI_Abort(mpicomm, -1);
    }
    if(m_num_visible_gpus < procs_per_node) {
      std::cerr << "NCCLCommunicator: rank " << rank() << ": has not enough GPUs available for given ranks\n";
      MPI_Abort(mpicomm, -2);
    }
    else{
      /// The number of GPUs on this node is greater than or equal to that of ranks assigned to this node;
      /// ensure that the right number of GPUs are used
      m_num_visible_gpus = procs_per_node;
    }

    // Assign GPUs to process
    int gpu_start, gpu_end;
    
    const int gpus_per_proc = m_num_visible_gpus / procs_per_node;
    const int num_leftover_gpus = m_num_visible_gpus % procs_per_node;
    gpu_start = rank_in_node * gpus_per_proc;
    gpu_end = (rank_in_node + 1) * gpus_per_proc;
    if(rank_in_node < num_leftover_gpus) {
      gpu_start += rank_in_node;
      gpu_end += rank_in_node + 1;
    }
    else {
      gpu_start += num_leftover_gpus;
      gpu_end += num_leftover_gpus;
    }

    // Construct GPU objects
    for(int gpu = gpu_start; gpu < gpu_end; ++gpu) {
      CUDACHECK(cudaSetDevice(gpu));
      m_gpus.push_back(gpu);
      m_streams.push_back(nullptr);

      CUDACHECK(cudaStreamCreate(&m_streams.back()));
    }

    // Get number of GPUs for current MPI rank
    m_num_gpus = m_gpus.size();
  }


  void nccl_setup() {

    if(m_num_gpus != 1){
      std::cerr << "NCCLCommunicator: rank " << rank() << ": the number of GPUs assigned to process is " << m_num_gpus << "; should be 1\n";
      MPI_Abort(mpicomm, -3);
    }

    /// Create nccl communicators
    int num_gpus_assigned = m_num_gpus;
    m_nccl_comm.resize(num_gpus_assigned);

    int nProcs = size();
    int myid = rank();
    int total_num_comms = nProcs*num_gpus_assigned;

    ncclUniqueId ncclId;
    if (myid == 0) {
      NCCLCHECK(ncclGetUniqueId(&ncclId));
    }

    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpicomm);

    if (nProcs == 1) {
      int gpuArray = 0;
      NCCLCHECK(ncclCommInitAll(&(m_nccl_comm[0]), 1, &gpuArray));
    }
    else {
      if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupStart());
      for(int i=0; i<num_gpus_assigned; i++){
        CUDACHECK(cudaSetDevice(m_gpus[i]));
        NCCLCHECK(ncclCommInitRank(&(m_nccl_comm[i]), total_num_comms, ncclId, num_gpus_assigned*myid+i));
      }
      if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupEnd());
    }
  } // nccl_setup

  void nccl_destroy() {
    int num_gpus_assigned = m_gpus.size();
    for(int i=0; i<num_gpus_assigned; i++){
      ncclCommDestroy(m_nccl_comm[i]);
    }
  }


 private:

  MPI_Comm mpicomm;

  ** List of GPU related variables.
  /// List of GPUs to be used
  std::vector<int> m_gpus;
  /// List of CUDA streams
  std::vector<cudaStream_t> m_streams;
  /// Number of GPUs allocated to the current rank
  int m_num_gpus;
  /// Number of visible GPUs on this compute node
  int m_num_visible_gpus;


  ** List of NCCL 2 related variables.
  /// NOTE: It is assumed that ONLY ONE GPU is allocated to one MPI rank
  std::vector<ncclComm_t> m_nccl_comm;
};
*/

class NCCLBackend {
 public:
  using algo_type = NCCLAllreduceAlgorithm;
  using comm_type = NCCLCommunicator;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type) {

    ncclDataType_t nccl_type;
    switch(sizeof(T)) {
      case 8:
        nccl_type = ncclDouble;
        break;
      case 4:
        nccl_type = ncclFloat;
        break;
      case 2:
        nccl_type = ncclHalf;
        break;
      default:
        throw_allreduce_exception("unsupported NCCL data type");
    }

    ncclRedOp_t nccl_redop;
    switch(op) {
      case ReductionOperator::sum:
        nccl_redop = ncclSum;
        break;
      case ReductionOperator::prod:
        nccl_redop = ncclProd;
        break;
      case ReductionOperator::min:
        nccl_redop = ncclMin;
        break;
      case ReductionOperator::max:
        nccl_redop = ncclMax;
        break;
      default:
        throw_allreduce_exception("unsupported NCCL reduction operator");
    }

    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    
    comm.Allreduce((void*) sendbuf, (void*) recvbuf, count, nccl_type, nccl_redop);
  }

  template <typename T>
  static void Allreduce(T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    Allreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingAllreduce(
      const T* sendbuf, T* recvbuf, size_t count,
      ReductionOperator op,
      comm_type& comm,
      AllreduceRequest& req,
      algo_type algo) {
    throw_allreduce_exception("Not implemented");
  }

  template <typename T>
  static void NonblockingAllreduce(
      T* recvbuf, size_t count,
      ReductionOperator op, comm_type& comm,
      AllreduceRequest& req,
      algo_type algo) {
    NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                         req, algo);
  }

};

}  // namespace allreduces


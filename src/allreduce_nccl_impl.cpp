#ifdef ALUMINUM_HAS_NCCL
#include "allreduce_nccl_impl.hpp"

namespace allreduces {

void NCCLCommunicator::mpi_setup(MPI_Comm comm_) {
    // Duplicate the communicator to avoid interference.
    MPI_Comm_dup(comm_, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &rank_in_comm);
    MPI_Comm_size(mpi_comm, &size_of_comm);
    MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &rank_in_local_comm);
    MPI_Comm_size(local_comm, &size_of_local_comm);
}
  
  
void NCCLCommunicator::gpu_setup() {

    const int rank_in_node = local_rank();
    const int procs_per_node = local_size();

    /// Determine number of visible GPUs on the current node
    CUDACHECK(cudaGetDeviceCount(&m_num_visible_gpus));
    if(m_num_visible_gpus < 1) {
      std::cerr << "NCCLCommunicator: rank " << rank() << ": has no GPUs found on the node\n";
      MPI_Abort(mpi_comm, -1);
    }
    if(m_num_visible_gpus < procs_per_node) {
      std::cerr << "NCCLCommunicator: rank " << rank() << ": has not enough GPUs available for given ranks\n";
      MPI_Abort(mpi_comm, -2);
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

void NCCLCommunicator::nccl_setup() {

    if(m_num_gpus != 1){
      std::cerr << "NCCLCommunicator: rank " << rank() << ": the number of GPUs assigned to process is " << m_num_gpus << "; should be 1\n";
      MPI_Abort(mpi_comm, -3);
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

    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpi_comm);

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

void NCCLCommunicator::nccl_destroy() {
    int num_gpus_assigned = m_num_gpus;
    //int num_gpus_assigned = m_gpus.size();
//printf("Am I here?: %d\n", num_gpus_assigned);
//printf("\tAm I here?: %d\n", m_nccl_comm.size());
    ncclCommDestroy(m_nccl_comm[0]);
    for(int i=0; i<num_gpus_assigned; i++){
      ncclCommDestroy(m_nccl_comm[i]);
    }
}

/// It is assumed that both sendbuf and recvbuf are in device memory
/// for NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
void NCCLCommunicator::Allreduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
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

}
#endif

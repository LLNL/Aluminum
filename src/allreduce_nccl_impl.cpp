#ifdef ALUMINUM_HAS_NCCL
#include "allreduce_nccl_impl.hpp"

namespace allreduces {

void NCCLCommunicator::gpu_setup() {
  int device;
  cudaGetDevice(&device);
  m_gpus.push_back(device);
  cudaStream_t s;
  cudaStreamCreate(&s);
  m_streams.push_back(s);
  m_num_gpus = 1;
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
  synchronize();
  for(int i=0; i<num_gpus_assigned; i++){
    ncclCommDestroy(m_nccl_comm[i]);
  }
}

void NCCLCommunicator::synchronize() {
  for (int i = 0; i < m_num_gpus; ++i) {
    cudaSetDevice(m_gpus[i]);
    cudaStreamSynchronize(m_streams[i]);
  }
}

cudaStream_t NCCLCommunicator::get_default_stream() {
  return m_streams[0];
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

} // namespace allreduces
#endif

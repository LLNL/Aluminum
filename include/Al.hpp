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

/**
 * @file
 * Aluminum initialization and communication operations.
 *
 * Aluminum provides an interface to high-performance and accelerator-aware
 * communication operations.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <mpi.h>

#include <Al_config.hpp>
#include "aluminum/base.hpp"
#include "aluminum/debug_helpers.hpp"
#include "aluminum/trace.hpp"

#if defined AL_HAS_CALIPER
#include <caliper/cali.h>
#include <caliper/cali_macros.h>
#endif

namespace Al {

/**
 * Initialize Aluminum.
 *
 * This must be called before any other calls to Aluminum are made,
 * except for Initialized(). It is safe to call this multiple times,
 * but it may not be called after Finalize().
 *
 * The \p argc and \p argv arguments are used to initialize MPI. They
 * may be null if the underlying MPI library does not rely on arguments
 * to initialize.
 *
 * This will initialize Aluminum to use the whole `MPI_COMM_WORLD`.
 * See Initialize(int&, char**&, MPI_Comm) if a specific subcommunicator
 * is desired.
 *
 * @param argc, argv The `argc` and `argv` arguments provided to the
 * binary.
 */
void Initialize(int& argc, char**& argv);
/**
 * Initialize Aluminum with an explicit MPI world communicator.
 *
 * This is identical to Initialize(int&, char**&), however, it allows
 * a different world communicator \p world_comm to be specified.
 * Aluminum will treat this as its world in instances where a default
 * world communicator is needed.
 *
 * Aluminum will create a duplicate of \p world_comm.
 *
 * @param argc, argv The `argc` and `argv` arguments provided to the
 * binary.
 * @param world_comm A default world communicator for Aluminum.
 */
void Initialize(int& argc, char**& argv, MPI_Comm world_comm);
/**
 * Clean up Aluminum.
 *
 * This will clean up all outstanding Aluminum resources and shut down
 * communication libraries.
 *
 * Do not make any additional calls to Aluminum after calling this
 * function, except for Initialized().
 */
void Finalize();
/**
 * Return true if Aluminum has been initialized and has not been finalized.
 *
 * It is always safe to call this.
 */
bool Initialized();

/**
 * Perform an allreduce.
 *
 * See \verbatim embed:rst:inline :ref:`Allreduce <allreduce>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced.
 * @param[out] recvbuf Buffer for the reduced vector.
 * @param[in] count Length of \p sendbuf and \p recvbuf in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce over.
 * @param[in] algo Request a particular allreduce algorithm.
 */
template <typename Backend, typename T>
void Allreduce(const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
               typename Backend::comm_type& comm,
               typename Backend::allreduce_algo_type algo =
                   Backend::allreduce_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_buffer(recvbuf, count);
  debug::check_overlap(sendbuf, count, recvbuf, count);
  AL_CALI_MARK_SCOPE("aluminum:Allreduce");
  internal::trace::record_op<Backend, T>("allreduce", comm, sendbuf, recvbuf,
                                         count);
  Backend::template Allreduce<T>(sendbuf, recvbuf, count, op, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Allreduce().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced. Will be replaced with the reduced
 * vector.
 * @param count Length of \p buffer in elements of type `T`.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce over.
 * @param algo Request a particular allreduce algorithm.
 */
template <typename Backend, typename T>
void Allreduce(T* buffer, size_t count, ReductionOperator op,
               typename Backend::comm_type& comm,
               typename Backend::allreduce_algo_type algo =
                   Backend::allreduce_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  AL_CALI_MARK_SCOPE("aluminum:Allreduce");
  internal::trace::record_op<Backend, T>("allreduce", comm, buffer, count);
  Backend::template Allreduce<T>(buffer, count, op, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Allreduce().
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced.
 * @param[out] recvbuf Buffer for the reduced vector.
 * @param[in] count Length of \p sendbuf and \p recvbuf in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular allreduce algorithm.
 */
template <typename Backend, typename T>
void NonblockingAllreduce(const T* sendbuf, T* recvbuf, size_t count,
                          ReductionOperator op,
                          typename Backend::comm_type& comm,
                          typename Backend::req_type& req,
                          typename Backend::allreduce_algo_type algo =
                              Backend::allreduce_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_buffer(recvbuf, count);
  debug::check_overlap(sendbuf, count, recvbuf, count);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAllreduce");
  internal::trace::record_op<Backend, T>("nonblocking-allreduce", comm, sendbuf,
                                         recvbuf, count);
  Backend::template NonblockingAllreduce<T>(sendbuf, recvbuf, count, op,
                                            comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Allreduce().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced. Will be replaced with the reduced
 * vector.
 * @param count Length of \p buffer in elements of type `T`.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce over.
 * @param[out] req Request object for the asynchronous operation.
 * @param algo Request a particular allreduce algorithm.
 */
template <typename Backend, typename T>
void NonblockingAllreduce(T* buffer, size_t count, ReductionOperator op,
                          typename Backend::comm_type& comm,
                          typename Backend::req_type& req,
                          typename Backend::allreduce_algo_type algo =
                              Backend::allreduce_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAllreduce");
  internal::trace::record_op<Backend, T>("nonblocking-allreduce", comm,
                                         buffer, count);
  Backend::template NonblockingAllreduce<T>(buffer, count, op,
                                            comm, req, algo);
}

/**
 * Perform a reduce-to-one.
 *
 * See \verbatim embed:rst:inline :ref:`Reduce <reduce>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced.
 * @param[out] recvbuf Buffer for the reduced vector.
 * @param[in] count Length of \p sendbuf and \p recvbuf in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator to reduce over.
 * @param[in] algo Request a particular reduction algorithm.
 */
template <typename Backend, typename T>
void Reduce(const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
            int root, typename Backend::comm_type& comm,
            typename Backend::reduce_algo_type algo =
                Backend::reduce_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(recvbuf, count, root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Reduce");
  internal::trace::record_op<Backend, T>("reduce", comm, sendbuf, recvbuf,
                                         count, root);
  Backend::template Reduce<T>(sendbuf, recvbuf, count, op, root, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Reduce().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced. Will be replaced with the reduced
 * vector.
 * @param[in] count Length of \p recvbuf in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator to reduce over.
 * @param[in] algo Request a particular reduction algorithm.
 */
template <typename Backend, typename T>
void Reduce(T* buffer, size_t count, ReductionOperator op, int root,
            typename Backend::comm_type& comm,
            typename Backend::reduce_algo_type algo =
                Backend::reduce_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Reduce");
  internal::trace::record_op<Backend, T>("reduce", comm, buffer, count, root);
  Backend::template Reduce<T>(buffer, count, op, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Reduce().
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced.
 * @param[out] recvbuf Buffer for the reduced vector.
 * @param[in] count Length of \p sendbuf and \p recvbuf in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator to reduce over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular reduction algorithm.
 */
template <typename Backend, typename T>
void NonblockingReduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, int root,
                       typename Backend::comm_type& comm,
                       typename Backend::req_type& req,
                       typename Backend::reduce_algo_type algo =
                           Backend::reduce_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(recvbuf, count, root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingReduce");
  internal::trace::record_op<Backend, T>("nonblocking-reduce", comm, sendbuf,
                                         recvbuf, count, root);
  Backend::template NonblockingReduce<T>(sendbuf, recvbuf, count, op, root, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Reduce().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced. Will be replaced with the reduced
 * vector.
 * @param[in] count Length of \p buffer in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator to reduce over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular reduction algorithm.
 */
template <typename Backend, typename T>
void NonblockingReduce(T* buffer, size_t count, ReductionOperator op, int root,
                       typename Backend::comm_type& comm,
                       typename Backend::req_type& req,
                       typename Backend::reduce_algo_type algo =
                           Backend::reduce_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingReduce");
  internal::trace::record_op<Backend, T>("nonblocking-reduce", comm, buffer,
                                         count, root);
  Backend::template NonblockingReduce<T>(buffer, count, op, root, comm, req, algo);
}

/**
 * Perform a reduce-scatter.
 *
 * See \verbatim embed:rst:inline :ref:`Reduce-scatter <reduce-scatter>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced/scattered.
 * @param[out] recvbuf Buffer for the scattered portion of the reduced vector.
 * @param[in] count Length of \p recvbuf in elements of type `T`.
 * \p sendbuf should be `count * comm.size()` elements.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[in] algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, typename Backend::comm_type& comm,
                    typename Backend::reduce_scatter_algo_type algo =
                        Backend::reduce_scatter_algo_type::automatic) {
  debug::check_buffer(sendbuf, count * comm.size());
  debug::check_buffer(recvbuf, count);
  debug::check_overlap(sendbuf, count * comm.size(), recvbuf, count);
  AL_CALI_MARK_SCOPE("aluminum:Reduce_scatter");
  internal::trace::record_op<Backend, T>("reduce_scatter", comm, sendbuf,
                                         recvbuf, count);
  Backend::template Reduce_scatter<T>(sendbuf, recvbuf, count, op, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Reduce_scatter().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced/scattered. Will be replaced with the
 * scattered portion of the reduced vector.
 * @param[in] count Length, in elements of type `T`, of the scattered
 * portion of the reduced vector. \p buffer should be `count * comm.size()`
 * elements.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[in] algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(T* buffer, size_t count, ReductionOperator op,
                    typename Backend::comm_type& comm,
                    typename Backend::reduce_scatter_algo_type algo =
                        Backend::reduce_scatter_algo_type::automatic) {
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:Reduce_scatter");
  internal::trace::record_op<Backend, T>("reduce_scatter", comm, buffer, count);
  Backend::template Reduce_scatter<T>(buffer, count, op, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Reduce_scatter().
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced/scattered.
 * @param[out] recvbuf Buffer for the scattered portion of the reduced vector.
 * @param[in] count Length of \p recvbuf in elements of type `T`.
 * \p sendbuf should be `count * comm.size()` elements.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
    typename Backend::comm_type& comm, typename Backend::req_type& req,
    typename Backend::reduce_scatter_algo_type algo =
        Backend::reduce_scatter_algo_type::automatic) {
  debug::check_buffer(sendbuf, count * comm.size());
  debug::check_buffer(recvbuf, count);
  debug::check_overlap(sendbuf, count * comm.size(), recvbuf, count);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingReduce_scatter");
  internal::trace::record_op<Backend, T>("nonblocking-reduce_scatter", comm,
                                         sendbuf, recvbuf, count);
  Backend::template NonblockingReduce_scatter<T>(
    sendbuf, recvbuf, count, op, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Reduce_scatter().
 *
 * @param[in,out] buffer Inout and output buffer initially containing
 * the local vector to be reduced/scattered. Will be replaced with the
 * scattered portion of the reduced vector.
 * @param[in] count Length, in elements of type `T`, of the scattered
 * portion of the reduced vector. \p buffer should be `count * comm.size()`
 * elements.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
    T* buffer, size_t count, ReductionOperator op,
    typename Backend::comm_type& comm, typename Backend::req_type& req,
    typename Backend::reduce_scatter_algo_type algo =
        Backend::reduce_scatter_algo_type::automatic) {
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:NonblockingReduce_scatter");
  internal::trace::record_op<Backend, T>("nonblocking-reduce_scatter", comm,
                                         buffer, count);
  Backend::template NonblockingReduce_scatter<T>(buffer, count, op, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`vector <comm-vector>` \endverbatim
 * Reduce_scatter().
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced/scattered.
 * @param[out] recvbuf Buffer for the scattered portion of the reduced vector.
 * @param[in] counts Vector of the length of the scattered vector each rank
 * should receive, in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[in] algo Request a particular reduce-scatterv algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatterv(const T* sendbuf, T* recvbuf, std::vector<size_t> counts,
                     ReductionOperator op, typename Backend::comm_type& comm,
                     typename Backend::reduce_scatterv_algo_type algo =
                         Backend::reduce_scatterv_algo_type::automatic) {
  debug::check_buffer(sendbuf, debug::sum(counts));
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_buffer(recvbuf, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_overlap(sendbuf, debug::sum(counts), recvbuf,
                       debug::get_rank_entry<Backend>(counts, comm));
  AL_CALI_MARK_SCOPE("aluminum:Reduce_scatterv");
  internal::trace::record_op<Backend, T>(
    "reduce_scatterv", comm, sendbuf, recvbuf, counts);
  Backend::template Reduce_scatterv<T>(sendbuf, recvbuf, counts,
                                       op, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Reduce_scatterv().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced/scattered. Will be replaced with the
 * scattered portion of the reduced vector.
 * @param[in] counts Vector of the length of the scattered vector each rank
 * should receive, in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[in] algo Request a particular reduce-scatterv algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatterv(T* buffer, std::vector<size_t> counts,
                     ReductionOperator op, typename Backend::comm_type& comm,
                     typename Backend::reduce_scatterv_algo_type algo =
                         Backend::reduce_scatterv_algo_type::automatic) {
  debug::check_buffer(buffer, debug::sum(counts));
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  AL_CALI_MARK_SCOPE("aluminum:Reduce_scatterv");
  internal::trace::record_op<Backend, T>(
    "reduce_scatterv", comm, buffer, counts);
  Backend::template Reduce_scatterv<T>(buffer, counts,
                                       op, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Reduce_scatterv().
 *
 * @param[in] sendbuf Buffer containing the local vector to be reduced/scattered.
 * @param[out] recvbuf Buffer for the scattered portion of the reduced vector.
 * @param[in] counts Vector of the length of the scattered vector each rank
 * should receive, in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular reduce-scatterv algorithm.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatterv(
    const T* sendbuf, T* recvbuf, std::vector<size_t> counts,
    ReductionOperator op, typename Backend::comm_type& comm,
    typename Backend::req_type& req,
    typename Backend::reduce_scatterv_algo_type algo =
        Backend::reduce_scatterv_algo_type::automatic) {
  debug::check_buffer(sendbuf, debug::sum(counts));
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_buffer(recvbuf, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_overlap(sendbuf, debug::sum(counts), recvbuf,
                       debug::get_rank_entry<Backend>(counts, comm));
  AL_CALI_MARK_SCOPE("aluminum:NonblockingReduce_scatterv");
  internal::trace::record_op<Backend, T>(
    "nonblocking-reduce_scatterv", comm, sendbuf, recvbuf, counts);
  Backend::template NonblockingReduce_scatterv<T>(
    sendbuf, recvbuf, counts, op, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Reduce_scatterv().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector to be reduced/scattered. Will be replaced with the
 * scattered portion of the reduced vector.
 * @param[in] counts Vector of the length of the scattered vector each rank
 * should receive, in elements of type `T`.
 * @param[in] op The reduction operation to perform.
 * @param[in] comm The communicator to reduce/scatter over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular reduce-scatterv algorithm.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatterv(
    T* buffer, std::vector<size_t> counts, ReductionOperator op,
    typename Backend::comm_type& comm, typename Backend::req_type& req,
    typename Backend::reduce_scatterv_algo_type algo =
        Backend::reduce_scatterv_algo_type::automatic) {
  debug::check_buffer(buffer, debug::sum(counts));
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingReduce_scatterv");
  internal::trace::record_op<Backend, T>(
    "nonblocking-reduce_scatterv", comm, buffer, counts);
  Backend::template NonblockingReduce_scatterv<T>(
    buffer, counts, op, comm, req, algo);
}

/**
 * Perform an allgather.
 *
 * See \verbatim embed:rst:inline :ref:`Allgather <allgather>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector.
 * @param[in] count Length of \p sendbuf in elements of type `T`.
 * \p recvbuf should be `count * comm.size()` elements.
 * @param[in] comm The communicator to allgather over.
 * @param[in] algo Request a particular allgather algorithm.
 */
template <typename Backend, typename T>
void Allgather(const T* sendbuf, T* recvbuf, size_t count,
               typename Backend::comm_type& comm,
               typename Backend::allgather_algo_type algo =
                   Backend::allgather_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_buffer(recvbuf, count * comm.size());
  debug::check_overlap(sendbuf, count, recvbuf, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:Allgather");
  internal::trace::record_op<Backend, T>("allgather", comm, sendbuf, recvbuf,
                                         count);
  Backend::template Allgather<T>(sendbuf, recvbuf, count, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Allgather().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local slice of data. Will contain the gathered vector.
 * @param[in] count Length, in elements of type `T`, of the local slice
 * of data. \p buffer should be `count * comm.size()` elements.
 * @param[in] comm The communicator to allgather over.
 * @param[in] algo Request a particular allgather algorithm.
 */
template <typename Backend, typename T>
void Allgather(T* buffer, size_t count, typename Backend::comm_type& comm,
               typename Backend::allgather_algo_type algo =
                   Backend::allgather_algo_type::automatic) {
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:Allgather");
  internal::trace::record_op<Backend, T>("allgather", comm, buffer, count);
  Backend::template Allgather<T>(buffer, count, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Allgather().
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector.
 * @param[in] count Length of \p sendbuf in elements of type `T`.
 * \p recvbuf should be `count * comm.size()` elements.
 * @param[in] comm The communicator to allgather over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular allgather algorithm.
 */
template <typename Backend, typename T>
void NonblockingAllgather(const T* sendbuf, T* recvbuf, size_t count,
                          typename Backend::comm_type& comm,
                          typename Backend::req_type& req,
                          typename Backend::allgather_algo_type algo =
                              Backend::allgather_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_buffer(recvbuf, count * comm.size());
  debug::check_overlap(sendbuf, count, recvbuf, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAllgather");
  internal::trace::record_op<Backend, T>("nonblocking-allgather", comm,
                                         sendbuf, recvbuf, count);
  Backend::template NonblockingAllgather<T>(sendbuf, recvbuf, count, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Allgather().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local slice of data. Will contain the gathered vector.
 * @param[in] count Length, in elements of type `T`, of the local slice
 * of data. \p buffer should be `count * comm.size()` elements.
 * @param[in] comm The communicator to allgather over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular allgather algorithm.
 */
template <typename Backend, typename T>
void NonblockingAllgather(T* buffer, size_t count,
                          typename Backend::comm_type& comm,
                          typename Backend::req_type& req,
                          typename Backend::allgather_algo_type algo =
                              Backend::allgather_algo_type::automatic) {
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAllgather");
  internal::trace::record_op<Backend, T>("nonblocking-allgather", comm,
                                         buffer, count);
  Backend::template NonblockingAllgather<T>(buffer, count, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`vector <comm-vector>` \endverbatim Allgather().
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector.
 * @param[in] counts Length of \p sendbuf on each rank in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank should be received.
 * @param[in] comm The communicator to allgatherv over.
 * @param[in] algo Request a particular allgatherv algorithm.
 */
template <typename Backend, typename T>
void Allgatherv(const T* sendbuf, T* recvbuf, std::vector<size_t> counts,
                std::vector<size_t> displs, typename Backend::comm_type& comm,
                typename Backend::allgatherv_algo_type algo =
                    Backend::allgatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_buffer(sendbuf, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_buffer(recvbuf, debug::sum(counts));
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  AL_CALI_MARK_SCOPE("aluminum:Allgatherv");
  internal::trace::record_op<Backend, T>("allgatherv", comm, sendbuf, recvbuf,
                                         counts, displs);
  Backend::template Allgatherv<T>(sendbuf, recvbuf, counts, displs, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Allgatherv().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local slice of data. Will contain the gathered vector.
 * @param[in] counts Length of each rank's slice in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p buffer
 * where data from the corresponding rank should be received.
 * @param[in] comm The communicator to allgatherv over.
 * @param[in] algo Request a particular allgatherv algorithm.
 */
template <typename Backend, typename T>
void Allgatherv(T* buffer, std::vector<size_t> counts,
                std::vector<size_t> displs, typename Backend::comm_type& comm,
                typename Backend::allgatherv_algo_type algo =
                    Backend::allgatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_buffer(buffer, debug::sum(counts));
  AL_CALI_MARK_SCOPE("aluminum:Allgatherv");
  internal::trace::record_op<Backend, T>("allgatherv", comm, buffer,
                                         counts, displs);
  Backend::template Allgatherv<T>(buffer, counts, displs, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Allgatherv().
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector.
 * @param[in] counts Length of \p sendbuf on each rank in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank should be received.
 * @param[in] comm The communicator to allgatherv over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular allgatherv algorithm.
 */
template <typename Backend, typename T>
void NonblockingAllgatherv(const T* sendbuf, T* recvbuf,
                           std::vector<size_t> counts,
                           std::vector<size_t> displs,
                           typename Backend::comm_type& comm,
                           typename Backend::req_type& req,
                           typename Backend::allgatherv_algo_type algo =
                               Backend::allgatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_buffer(sendbuf, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_buffer(recvbuf, debug::sum(counts));
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAllgatherv");
  internal::trace::record_op<Backend, T>("nonblocking-allgatherv", comm,
                                         sendbuf, recvbuf, counts, displs);
  Backend::template NonblockingAllgatherv<T>(sendbuf, recvbuf, counts, displs, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Allgatherv().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local slice of data. Will contain the gathered vector.
 * @param[in] counts Length of each rank's slice in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p buffer
 * where data from the corresponding rank should be received.
 * @param[in] comm The communicator to allgatherv over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular allgatherv algorithm.
 */
template <typename Backend, typename T>
void NonblockingAllgatherv(T* buffer, std::vector<size_t> counts,
                           std::vector<size_t> displs,
                           typename Backend::comm_type& comm,
                           typename Backend::req_type& req,
                           typename Backend::allgatherv_algo_type algo =
                               Backend::allgatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_buffer(buffer, debug::sum(counts));
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAllgatherv");
  internal::trace::record_op<Backend, T>("nonblocking-allgatherv", comm,
                                         buffer, counts, displs);
  Backend::template NonblockingAllgatherv<T>(buffer, counts, displs, comm, req, algo);
}

/**
 * Perform a barrier synchronization.
 *
 * See \verbatim embed:rst:inline :ref:`Barrier <barrier>`. \endverbatim
 *
 * @param[in] comm The communicator to synchronize over.
 * @param[in] algo Request a particular barrier algorithm.
 */
template <typename Backend>
void Barrier(typename Backend::comm_type& comm,
             typename Backend::barrier_algo_type algo =
             Backend::barrier_algo_type::automatic) {
  AL_CALI_MARK_SCOPE("aluminum:Barrier");
  internal::trace::record_op<Backend, void>("barrier", comm);
  Backend::Barrier(comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Barrier().
 *
 * @param[in] comm The communicator to synchronize over.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular barrier algorithm.
 */
template <typename Backend>
void NonblockingBarrier(typename Backend::comm_type& comm,
                        typename Backend::req_type& req,
                        typename Backend::barrier_algo_type algo =
                        Backend::barrier_algo_type::automatic) {
  AL_CALI_MARK_SCOPE("aluminum:NonblockingBarrier");
  internal::trace::record_op<Backend, void>("nonblocking-barrier", comm);
  Backend::NonblockingBarrier(comm, req, algo);
}

/**
 * Perform a broadcast.
 *
 * Broadcast is always \verbatim embed:rst:inline :ref:`in-place <comm-inplace>`. \endverbatim
 *
 * See \verbatim embed:rst:inline :ref:`Bcast <bcast>`. \endverbatim
 *
 * @param[in,out] buffer On the root, buffer containing the data to
 * broadcast. On other ranks, buffer that will receive the broadcasted data.
 * @param[in] count Length of \p buffer in elements of type `T`.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator to broadcast over.
 * @param[in] algo Request a particular broadcast algorithm.
 */
template <typename Backend, typename T>
void Bcast(T* buffer, size_t count, int root, typename Backend::comm_type& comm,
           typename Backend::bcast_algo_type algo =
               Backend::bcast_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Bcast");
  internal::trace::record_op<Backend, T>("bcast", comm, buffer, count, root);
  Backend::template Bcast<T>(buffer, count, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Bcast().
 *
 * Broadcast is always \verbatim embed:rst:inline :ref:`in-place <comm-inplace>`. \endverbatim
 *
 * @param[in,out] buffer On the root, buffer containing the data to
 * broadcast. On other ranks, buffer that will receive the broadcasted data.
 * @param[in] count Length of \p buffer in elements of type `T`.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator to broadcast over.
 * @param[out] req Request for the asynchronous operation.
 * @param[in] algo Request a particular broadcast algorithm.
 */
template <typename Backend, typename T>
void NonblockingBcast(T* buffer, size_t count, int root,
                      typename Backend::comm_type& comm,
                      typename Backend::req_type& req,
                      typename Backend::bcast_algo_type algo =
                          Backend::bcast_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingBcast");
  internal::trace::record_op<Backend, T>("nonblocking-bcast", comm, buffer,
                                         count, root);
  Backend::template NonblockingBcast<T>(buffer, count, root, comm, req, algo);
}

/**
 * Perform an all-to-all.
 *
 * See \verbatim embed:rst:inline :ref:`Alltoall <alltoall>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local vector slices.
 * @param[out] recvbuf Buffer for the assembled slices.
 * @param[in] count Length of each slice in \p sendbuf in elements of type `T`.
 * \p sendbuf and \p recvbuf should be `count * comm.size()` elements.
 * @param[in] comm The communicator for this all-to-all operation.
 * @param[in] algo Request a particular all-to-all algorithm.
 */
template <typename Backend, typename T>
void Alltoall(const T* sendbuf, T* recvbuf, size_t count,
              typename Backend::comm_type& comm,
              typename Backend::alltoall_algo_type algo =
                  Backend::alltoall_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_buffer(recvbuf, count * comm.size());
  debug::check_overlap(sendbuf, count, recvbuf, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:Alltoall");
  internal::trace::record_op<Backend, T>("alltoall", comm, sendbuf, recvbuf,
                                         count);
  Backend::template Alltoall<T>(sendbuf, recvbuf, count, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Alltoall().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector slices. Will be replaced with the assembled slices.
 * @param[in] count Length of each slice in \p sendbuf in elements of type `T`.
 * \p buffer should be `count * comm.size()` elements.
 * @param[in] comm The communicator fo this all-to-all operation.
 * @param[in] algo Request a particular all-to-all algorithm.
 */
template <typename Backend, typename T>
void Alltoall(T* buffer, size_t count, typename Backend::comm_type& comm,
              typename Backend::alltoall_algo_type algo =
                  Backend::alltoall_algo_type::automatic) {
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:Alltoall");
  internal::trace::record_op<Backend, T>("alltoall", comm, buffer, count);
  Backend::template Alltoall<T>(buffer, count, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`nonblocking <comm-nonblocking>` \endverbatim Alltoall().
 *
 * @param[in] sendbuf Buffer containing the local vector slices.
 * @param[out] recvbuf Buffer for the assembled slices.
 * @param[in] count Length of each slice in \p sendbuf in elements of type `T`.
 * \p sendbuf and \p recvbuf should be `count * comm.size()` elements.
 * @param[in] comm The communicator for this all-to-all operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular all-to-all algorithm.
 */
template <typename Backend, typename T>
void NonblockingAlltoall(const T* sendbuf, T* recvbuf, size_t count,
                         typename Backend::comm_type& comm,
                         typename Backend::req_type& req,
                         typename Backend::alltoall_algo_type algo =
                             Backend::alltoall_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_buffer(recvbuf, count * comm.size());
  debug::check_overlap(sendbuf, count, recvbuf, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAlltoall");
  internal::trace::record_op<Backend, T>("nonblocking-alltoall", comm, sendbuf,
                                         recvbuf, count);
  Backend::template NonblockingAlltoall<T>(sendbuf, recvbuf, count,
                                           comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Alltoall().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector slices. Will be replaced with the assembled slices.
 * @param[in] count Length of each slice in \p sendbuf in elements of type `T`.
 * \p buffer should be `count * comm.size()` elements.
 * @param[in] comm The communicator fo this all-to-all operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular all-to-all algorithm.
 */
template <typename Backend, typename T>
void NonblockingAlltoall(T* buffer, size_t count,
                         typename Backend::comm_type& comm,
                         typename Backend::req_type& req,
                         typename Backend::alltoall_algo_type algo =
                             Backend::alltoall_algo_type::automatic) {
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAlltoall");
  internal::trace::record_op<Backend, T>("nonblocking-alltoall", comm, buffer,
                                         count);
  Backend::template NonblockingAlltoall<T>(buffer, count, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`vector <comm-vector>` \endverbatim Alltoall().
 *
 * @param[in] sendbuf Buffer containing the local vector slices.
 * @param[in] send_counts Length of each slice in \p sendbuf in elements of type `T`.
 * @param[in] send_displs Offsets, in elements of type `T`, into \p sendbuf
 * where the data for the corresponding rank begins.
 * @param[out] recvbuf Buffer for the assembled slices.
 * @param[in] recv_counts Length of each slice that will be received in
 * \p recvbuf in elements of type `T`.
 * @param[in] recv_displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank should be received.
 * @param[in] comm Communicator for this all-to-all operation.
 * @param[in] algo Request a particular vector all-to-all algorithm.
 */
template <typename Backend, typename T>
void Alltoallv(const T* sendbuf, std::vector<size_t> send_counts,
               std::vector<size_t> send_displs, T* recvbuf,
               std::vector<size_t> recv_counts, std::vector<size_t> recv_displs,
               typename Backend::comm_type& comm,
               typename Backend::alltoallv_algo_type algo =
                   Backend::alltoallv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(send_counts, comm);
  debug::check_vector_is_comm_sized<Backend>(send_displs, comm);
  debug::check_buffer(sendbuf, debug::sum(send_counts));
  debug::check_vector_is_comm_sized<Backend>(recv_counts, comm);
  debug::check_vector_is_comm_sized<Backend>(recv_displs, comm);
  debug::check_buffer(recvbuf, debug::sum(recv_counts));
  AL_CALI_MARK_SCOPE("aluminum:Alltoallv");
  internal::trace::record_op<Backend, T>(
    "alltoallv", comm,
    sendbuf, send_counts, send_displs,
    recvbuf, recv_counts, recv_displs);
  Backend::template Alltoallv<T>(sendbuf, send_counts, send_displs,
                                 recvbuf, recv_counts, recv_displs,
                                 comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Alltoallv().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector slices for each rank. Will contain the assembled
 * slices for this rank.
 * @param[in] counts Length of the slice sent to and received from each
 * rank, in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p buffer
 * for data sent to and received from the corresponding rank.
 * @param[in] comm Communicator for this all-to-all operation.
 * @param[in] algo Request a particular vector all-to-all algorithm.
 */
template <typename Backend, typename T>
void Alltoallv(T* buffer, std::vector<size_t> counts,
               std::vector<size_t> displs, typename Backend::comm_type& comm,
               typename Backend::alltoallv_algo_type algo =
                   Backend::alltoallv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(buffer, debug::sum(counts));
  AL_CALI_MARK_SCOPE("aluminum:Alltoallv");
  internal::trace::record_op<Backend, T>(
    "alltoallv", comm, buffer, counts, displs);
  Backend::template Alltoallv<T>(buffer, counts, displs, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Alltoallv().
 *
 * @param[in] sendbuf Buffer containing the local vector slices.
 * @param[in] send_counts Length of each slice in \p sendbuf in elements of type `T`.
 * @param[in] send_displs Offsets, in elements of type `T`, into \p sendbuf
 * where the data for the corresponding rank begins.
 * @param[out] recvbuf Buffer for the assembled slices.
 * @param[in] recv_counts Length of each slice that will be received in
 * \p recvbuf in elements of type `T`.
 * @param[in] recv_displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank should be received.
 * @param[in] comm Communicator for this all-to-all operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular vector all-to-all algorithm.
 */
template <typename Backend, typename T>
void NonblockingAlltoallv(const T* sendbuf, std::vector<size_t> send_counts,
                          std::vector<size_t> send_displs, T* recvbuf,
                          std::vector<size_t> recv_counts,
                          std::vector<size_t> recv_displs,
                          typename Backend::comm_type& comm,
                          typename Backend::req_type& req,
                          typename Backend::alltoallv_algo_type algo =
                              Backend::alltoallv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(send_counts, comm);
  debug::check_vector_is_comm_sized<Backend>(send_displs, comm);
  debug::check_buffer(sendbuf, debug::sum(send_counts));
  debug::check_vector_is_comm_sized<Backend>(recv_counts, comm);
  debug::check_vector_is_comm_sized<Backend>(recv_displs, comm);
  debug::check_buffer(recvbuf, debug::sum(recv_counts));
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAlltoallv");
  internal::trace::record_op<Backend, T>(
    "nonblocking-alltoallv", comm,
    sendbuf, send_counts, send_displs,
    recvbuf, recv_counts, recv_displs);
  Backend::template NonblockingAlltoallv<T>(
    sendbuf, send_counts, send_displs,
    recvbuf, recv_counts, recv_displs,
    comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Alltoallv().
 *
 * @param[in,out] buffer Input and output buffer initially containing
 * the local vector slices for each rank. Will contain the assembled
 * slices for this rank.
 * @param[in] counts Length of the slice sent to and received from each
 * rank, in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p buffer
 * for data sent to and received from the corresponding rank.
 * @param[in] comm Communicator for this all-to-all operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular vector all-to-all algorithm.
 */
template <typename Backend, typename T>
void NonblockingAlltoallv(T* buffer, std::vector<size_t> counts,
                          std::vector<size_t> displs,
                          typename Backend::comm_type& comm,
                          typename Backend::req_type& req,
                          typename Backend::alltoallv_algo_type algo =
                              Backend::alltoallv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(buffer, debug::sum(counts));
  AL_CALI_MARK_SCOPE("aluminum:NonblockingAlltoallv");
  internal::trace::record_op<Backend, T>(
    "nonblocking-alltoallv", comm, buffer, counts, displs);
  Backend::template NonblockingAlltoallv<T>(buffer, counts, displs, comm,
                                            req, algo);
}

/**
 * Perform a gather-to-one.
 *
 * See \verbatim embed:rst:inline :ref:`Gather <gather>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector on the root.
 * @param[in] count Length of each local slice in elements of type `T`.
 * \p sendbuf should be `count` elements and \p recvbuf should be
 * `count * comm.size()` elements on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this gather operation.
 * @param[in] algo Request a particular gather algorithm.
 */
template <typename Backend, typename T>
void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
            typename Backend::comm_type& comm,
            typename Backend::gather_algo_type algo =
                Backend::gather_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(recvbuf, count, root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Gather");
  internal::trace::record_op<Backend, T>("gather", comm, sendbuf, recvbuf,
                                         count, root);
  Backend::template Gather<T>(sendbuf, recvbuf, count, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Gather().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * local slice of data. On the root, its slice must be in the location
 * corresponding to its rank position. On non-roots, the entire buffer
 * is the slice. Will be replaced with the gathered vector on the root.
 * @param[in] count Length of each local slice in elements of type `T`.
 * \p buffer should be `count` elements on non-roots and `count * comm.size()`
 * elements on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator fo this gather operation.
 * @param[in] algo Request a particular gather algorithm.
 */
template <typename Backend, typename T>
void Gather(T* buffer, size_t count, int root,
            typename Backend::comm_type& comm,
            typename Backend::gather_algo_type algo =
                Backend::gather_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Gather");
  internal::trace::record_op<Backend, T>("gather", comm, buffer, count, root);
  Backend::template Gather<T>(buffer, count, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Gather().
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector on the root.
 * @param[in] count Length of each local slice in elements of type `T`.
 * \p sendbuf should be `count` elements and \p recvbuf should be
 * `count * comm.size()` elements on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this gather operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular gather algorithm.
 */
template <typename Backend, typename T>
void NonblockingGather(const T* sendbuf, T* recvbuf, size_t count, int root,
                       typename Backend::comm_type& comm,
                       typename Backend::req_type& req,
                       typename Backend::gather_algo_type algo =
                           Backend::gather_algo_type::automatic) {
  debug::check_buffer(sendbuf, count);
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(recvbuf, count, root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingGather");
  internal::trace::record_op<Backend, T>("nonblocking-gather", comm, sendbuf,
                                         recvbuf, count, root);
  Backend::template NonblockingGather<T>(sendbuf, recvbuf, count, root,
                                         comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Gather().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * local slice of data. On the root, its slice must be in the location
 * corresponding to its rank position. On non-roots, the entire buffer
 * is the slice. Will be replaced with the gathered vector on the root.
 * @param[in] count Length of each local slice in elements of type `T`.
 * \p buffer should be `count` elements on non-roots and `count * comm.size()`
 * elements on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator fo this gather operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular gather algorithm.
 */
template <typename Backend, typename T>
void NonblockingGather(T* buffer, size_t count, int root,
                       typename Backend::comm_type& comm,
                       typename Backend::req_type& req,
                       typename Backend::gather_algo_type algo =
                           Backend::gather_algo_type::automatic) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingGather");
  internal::trace::record_op<Backend, T>("nonblocking-gather", comm, buffer,
                                         count, root);
  Backend::template NonblockingGather<T>(buffer, count, root, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`vector <comm-vector>` \endverbatim Gather().
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector on the root.
 * @param[in] counts Length of each rank's slice in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank will be received on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this gather operation.
 * @param[in] algo Request a particular vector gather algorithm.
 */
template <typename Backend, typename T>
void Gatherv(const T* sendbuf, T* recvbuf, std::vector<size_t> counts,
             std::vector<size_t> displs, int root,
             typename Backend::comm_type& comm,
             typename Backend::gatherv_algo_type algo =
                 Backend::gatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(sendbuf, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(recvbuf, debug::sum(counts), root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Gatherv");
  internal::trace::record_op<Backend, T>("gatherv", comm, sendbuf, recvbuf,
                                         counts, displs, root);
  Backend::template Gatherv<T>(sendbuf, recvbuf, counts, displs, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Gatherv().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * local slice of data. On the root, its slice must be in the location
 * corresponding to its rank position. On non-roots, the entire buffer
 * is the slice. Will be replaced with the gathered vector on the root.
 * @param[in] counts Length of each rank's slice in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank will be received on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this gather operation.
 * @param[in] algo Request a particular vector gather algorithm.
 */
template <typename Backend, typename T>
void Gatherv(T* buffer, std::vector<size_t> counts, std::vector<size_t> displs,
             int root, typename Backend::comm_type& comm,
             typename Backend::gatherv_algo_type algo =
                 Backend::gatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(buffer, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:Gatherv");
  internal::trace::record_op<Backend, T>("gatherv", comm, buffer,
                                         counts, displs, root);
  Backend::template Gatherv<T>(buffer, counts, displs, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Gatherv().
 *
 * @param[in] sendbuf Buffer containing the local slice of data.
 * @param[out] recvbuf Buffer for the gathered vector on the root.
 * @param[in] counts Length of each rank's slice in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank will be received on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this gather operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular vector gather algorithm.
 */
template <typename Backend, typename T>
void NonblockingGatherv(const T* sendbuf, T* recvbuf,
                        std::vector<size_t> counts, std::vector<size_t> displs,
                        int root, typename Backend::comm_type& comm,
                        typename Backend::req_type& req,
                        typename Backend::gatherv_algo_type algo =
                            Backend::gatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(sendbuf, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(recvbuf, debug::sum(counts), root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingGatherv");
  internal::trace::record_op<Backend, T>("nonblocking-gatherv",
                                         comm, sendbuf, recvbuf,
                                         counts, displs, root);
  Backend::template NonblockingGatherv<T>(
    sendbuf, recvbuf, counts, displs, root, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Gatherv().
 *
 * @param[in,out] buffer Inout and output buffer initially containing the
 * local slice of data. On the root, its slice must be in the location
 * corresponding to its rank position. On non-roots, the entire buffer
 * is the slice. Will be replaced with the gathered vector on the root.
 * @param[in] counts Length of each rank's slice in elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p recvbuf
 * where data from the corresponding rank will be received on the root.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this gather operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular vector gather algorithm.
 */
template <typename Backend, typename T>
void NonblockingGatherv(T* buffer, std::vector<size_t> counts,
                        std::vector<size_t> displs, int root,
                        typename Backend::comm_type& comm,
                        typename Backend::req_type& req,
                        typename Backend::gatherv_algo_type algo =
                            Backend::gatherv_algo_type::automatic) {
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(buffer, debug::get_rank_entry<Backend>(counts, comm));
  debug::check_rank<Backend>(root, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingGatherv");
  internal::trace::record_op<Backend, T>("nonblocking-gatherv",
                                         comm, buffer,
                                         counts, displs, root);
  Backend::template NonblockingGatherv<T>(
    buffer, counts, displs, root, comm, req, algo);
}

/**
 * Perform a scatter-to-all.
 *
 * See \verbatim embed:rst:inline :ref:`Scatter <scatter>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the complete vector at the root.
 * Empty on non-roots.
 * @param[out] recvbuf Buffer for the scattered slice.
 * @param[in] count Length of each scattered slice in elements of type `T`.
 * \p sendbuf should be `count * comm.size()` elements on the root and
 * empty on non-roots. \p recvbuf should be `count` elements.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[in] algo Request a particular scatter algorithm.
 */
template <typename Backend, typename T>
void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
             typename Backend::comm_type& comm,
             typename Backend::scatter_algo_type algo =
                 Backend::scatter_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(sendbuf, count * comm.size(), root, comm);
  debug::check_buffer(recvbuf, count);
  AL_CALI_MARK_SCOPE("aluminum:Scatter");
  internal::trace::record_op<Backend, T>("scatter", comm, sendbuf, recvbuf,
                                         count, root);
  Backend::template Scatter<T>(sendbuf, recvbuf, count, root, comm, algo);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Scatter().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * complete vector at the root and empty on non-roots. Will be replaced
 * with the scattered slice on each rank. At the root, the scattered
 * slice is in its corresponding rank position.
 * @param[in] count Length of each scattered slice in elements of type `T`.
 * \p buffer should be `count * comm.size()` elements on the root and
 * `count` elements on non-roots.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[in] algo Request a particular scatter algorithm.
 */
template <typename Backend, typename T>
void Scatter(T* buffer, size_t count, int root,
             typename Backend::comm_type& comm,
             typename Backend::scatter_algo_type algo =
                 Backend::scatter_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:Scatter");
  internal::trace::record_op<Backend, T>("scatter", comm, buffer, count, root);
  Backend::template Scatter<T>(buffer, count, root, comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Scatter().
 *
 * @param[in] sendbuf Buffer containing the complete vector at the root.
 * Empty on non-roots.
 * @param[out] recvbuf Buffer for the scattered slice.
 * @param[in] count Length of each scattered slice in elements of type `T`.
 * \p sendbuf should be `count * comm.size()` elements on the root and
 * empty on non-roots. \p recvbuf should be `count` elements.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular scatter algorithm.
 */
template <typename Backend, typename T>
void NonblockingScatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                        typename Backend::comm_type& comm,
                        typename Backend::req_type& req,
                        typename Backend::scatter_algo_type algo =
                            Backend::scatter_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer_root<Backend>(sendbuf, count * comm.size(), root, comm);
  debug::check_buffer(recvbuf, count);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingScatter");
  internal::trace::record_op<Backend, T>("nonblocking-scatter", comm, sendbuf,
                                         recvbuf, count, root);
  Backend::template NonblockingScatter<T>(sendbuf, recvbuf, count, root,
                                         comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Scatter().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * complete vector at the root and empty on non-roots. Will be replaced
 * with the scattered slice on each rank. At the root, the scattered
 * slice is in its corresponding rank position.
 * @param[in] count Length of each scattered slice in elements of type `T`.
 * \p buffer should be `count * comm.size()` elements on the root and
 * `count` elements on non-roots.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular scatter algorithm.
 */
template <typename Backend, typename T>
void NonblockingScatter(T* buffer, size_t count, int root,
                        typename Backend::comm_type& comm,
                        typename Backend::req_type& req,
                        typename Backend::scatter_algo_type algo =
                            Backend::scatter_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_buffer(buffer, count * comm.size());
  AL_CALI_MARK_SCOPE("aluminum:NonblockingScatter");
  internal::trace::record_op<Backend, T>("nonblocking-scatter", comm, buffer,
                                         count, root);
  Backend::template NonblockingScatter<T>(buffer, count, root, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`vector <comm-vector>` \endverbatim Scatter().
 *
 * @param[in] sendbuf Buffer containing the complete vector at the root.
 * Empty on non-roots.
 * @param[out] recvbuf Buffer for the scattered slice.
 * @param[in] counts Length of the slice each rank will receive, in
 * elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p sendbuf
 * where the slices for each corresponding rank begin.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[in] algo Request a particular vector scatter algorithm.
 */
template <typename Backend, typename T>
void Scatterv(const T* sendbuf, T* recvbuf, std::vector<size_t> counts,
              std::vector<size_t> displs, int root,
              typename Backend::comm_type& comm,
              typename Backend::scatterv_algo_type algo =
                  Backend::scatterv_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer_root<Backend>(sendbuf, debug::sum(counts), root, comm);
  debug::check_buffer(recvbuf, debug::get_rank_entry<Backend>(counts, comm));
  AL_CALI_MARK_SCOPE("aluminum:Scatterv");
  internal::trace::record_op<Backend, T>("scatterv", comm,
                                         sendbuf, recvbuf,
                                         counts, displs, root);
  Backend::template Scatterv<T>(sendbuf, recvbuf, counts, displs, root,
                                comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Scatterv().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * complete vector at the root and empty on non-roots. Will be replaced
 * with the scattered slice on each rank. At the root, the scattered
 * slice is in its corresponding rank position.
 * @param[in] counts Length of the slice each rank will receive, in
 * elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p buffer
 * where the slices for each corresponding rank begin.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[in] algo Request a particular vector scatter algorithm.
 */
template <typename Backend, typename T>
void Scatterv(T* buffer, std::vector<size_t> counts, std::vector<size_t> displs,
              int root, typename Backend::comm_type& comm,
              typename Backend::scatterv_algo_type algo =
                  Backend::scatterv_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(buffer, debug::get_rank_entry<Backend>(counts, comm));
  AL_CALI_MARK_SCOPE("aluminum:Scatterv");
  internal::trace::record_op<Backend, T>("scatterv", comm,
                                         buffer, counts, displs, root);
  Backend::template Scatterv<T>(buffer, counts, displs, root,
                                comm, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Scatterv().
 *
 * @param[in] sendbuf Buffer containing the complete vector at the root.
 * Empty on non-roots.
 * @param[out] recvbuf Buffer for the scattered slice.
 * @param[in] counts Length of the slice each rank will receive, in
 * elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p sendbuf
 * where the slices for each corresponding rank begin.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[out] req Request object for the asynchronus operation.
 * @param[in] algo Request a particular vector scatter algorithm.
 */
template <typename Backend, typename T>
void NonblockingScatterv(const T* sendbuf, T* recvbuf,
                         std::vector<size_t> counts, std::vector<size_t> displs,
                         int root, typename Backend::comm_type& comm,
                         typename Backend::req_type& req,
                         typename Backend::scatterv_algo_type algo =
                             Backend::scatterv_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer_root<Backend>(sendbuf, debug::sum(counts), root, comm);
  debug::check_buffer(recvbuf, debug::get_rank_entry<Backend>(counts, comm));
  AL_CALI_MARK_SCOPE("aluminum:NonblockingScatterv");
  internal::trace::record_op<Backend, T>("nonblocking-scatterv", comm,
                                         sendbuf, recvbuf,
                                         counts, displs, root);
  Backend::template NonblockingScatterv<T>(
    sendbuf, recvbuf, counts, displs, root, comm, req, algo);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim Scatterv().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * complete vector at the root and empty on non-roots. Will be replaced
 * with the scattered slice on each rank. At the root, the scattered
 * slice is in its corresponding rank position.
 * @param[in] counts Length of the slice each rank will receive, in
 * elements of type `T`.
 * @param[in] displs Offsets, in elements of type `T`, into \p buffer
 * where the slices for each corresponding rank begin.
 * @param[in] root Root rank for the operation.
 * @param[in] comm The communicator for this scatter operation.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] algo Request a particular vector scatter algorithm.
 */
template <typename Backend, typename T>
void NonblockingScatterv(T* buffer, std::vector<size_t> counts,
                         std::vector<size_t> displs, int root,
                         typename Backend::comm_type& comm,
                         typename Backend::req_type& req,
                         typename Backend::scatterv_algo_type algo =
                             Backend::scatterv_algo_type::automatic) {
  debug::check_rank<Backend>(root, comm);
  debug::check_vector_is_comm_sized<Backend>(counts, comm);
  debug::check_vector_is_comm_sized<Backend>(displs, comm);
  debug::check_buffer(buffer, debug::get_rank_entry<Backend>(counts, comm));
  AL_CALI_MARK_SCOPE("aluminum:NonblockingScatterv");
  internal::trace::record_op<Backend, T>("nonblocking-scatterv", comm,
                                         buffer, counts, displs, root);
  Backend::template NonblockingScatterv<T>(
    buffer, counts, displs, root, comm, req, algo);
}

/**
 * Send a point-to-point message.
 *
 * See \verbatim embed:rst:inline :ref:`Send and Recv <send-and-recv>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local data to send.
 * @param[in] count Length of \p sendbuf in elements of type `T`.
 * @param[in] dest Rank in comm to send to.
 * @param[in] comm Communicator to send within.
 */
template <typename Backend, typename T>
void Send(const T* sendbuf, size_t count, int dest,
          typename Backend::comm_type& comm) {
  debug::check_buffer(sendbuf, count);
  debug::check_rank<Backend>(dest, comm);
  AL_CALI_MARK_SCOPE("aluminum:Send");
  internal::trace::record_op<Backend, T>("send", comm, sendbuf, count, dest);
  Backend::template Send<T>(sendbuf, count, dest, comm);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Send().
 *
 * @param[in] sendbuf Buffer containing the local data to send.
 * @param[in] count Length of \p sendbuf in elements of type `T`.
 * @param[in] dest Rank in comm to send to.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] comm Communicator to send within.
 */
template <typename Backend, typename T>
void NonblockingSend(const T* sendbuf, size_t count, int dest,
                     typename Backend::comm_type& comm,
                     typename Backend::req_type& req) {
  debug::check_buffer(sendbuf, count);
  debug::check_rank<Backend>(dest, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingSend");
  internal::trace::record_op<Backend, T>("nonblocking-send", comm, sendbuf,
                                         count, dest);
  Backend::template NonblockingSend<T>(sendbuf, count, dest, comm, req);
}

/**
 * Receive a point-to-point message.
 *
 * See \verbatim embed:rst:inline :ref:`Send and Recv <send-and-recv>`. \endverbatim
 *
 * @param[out] recvbuf Buffer to receive the sent data.
 * @param[in] count Length of \p recvbuf in elements of type `T`.
 * @param[in] src Rank in comm to receive from.
 * @param[in] comm Communicator to receive within.
 */
template <typename Backend, typename T>
void Recv(T* recvbuf, size_t count, int src,
          typename Backend::comm_type& comm) {
  debug::check_buffer(recvbuf, count);
  debug::check_rank<Backend>(src, comm);
  AL_CALI_MARK_SCOPE("aluminum:Recv");
  internal::trace::record_op<Backend, T>("recv", comm, recvbuf, count, src);
  Backend::template Recv<T>(recvbuf, count, src, comm);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim Recv().
 *
 * @param[out] recvbuf Buffer to receive the sent data.
 * @param[in] count Length of \p recvbuf in elements of type `T`.
 * @param[in] src Rank in comm to receive from.
 * @param[out] req Request object for the asynchronous operation.
 * @param[in] comm Communicator to receive within.
 */
template <typename Backend, typename T>
void NonblockingRecv(T* recvbuf, size_t count, int src,
                     typename Backend::comm_type& comm,
                     typename Backend::req_type& req) {
  debug::check_buffer(recvbuf, count);
  debug::check_rank<Backend>(src, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingRecv");
  internal::trace::record_op<Backend, T>("nonblocking-recv", comm, recvbuf,
                                         count, src);
  Backend::template NonblockingRecv<T>(recvbuf, count, src, comm, req);
}

/**
 * Perform a simultaneous Send() and Recv().
 *
 * See \verbatim embed:rst:inline :ref:`SendRecv <sendrecv>`. \endverbatim
 *
 * @param[in] sendbuf Buffer containing the local data to send.
 * @param[in] send_count Length of \p sendbuf in elements of type `T`.
 * @param[in] dest Rank in comm to send to.
 * @param[out] recvbuf Buffer to receive the sent data.
 * @param[in] recv_count Length of \p recvbuf in elements of type `T`.
 * @param[in] src Rank in comm to receive from.
 * @param[in] comm Communicator to send/recv within.
 */
template <typename Backend, typename T>
void SendRecv(const T* sendbuf, size_t send_count, int dest, T* recvbuf,
              size_t recv_count, int src, typename Backend::comm_type& comm) {
  debug::check_buffer(sendbuf, send_count);
  debug::check_buffer(recvbuf, recv_count);
  debug::check_overlap(sendbuf, send_count, recvbuf, recv_count);
  debug::check_rank<Backend>(dest, comm);
  debug::check_rank<Backend>(src, comm);
  AL_CALI_MARK_SCOPE("aluminum:SendRecv");
  internal::trace::record_op<Backend, T>("sendrecv", comm, sendbuf, send_count,
                                         dest, recvbuf, recv_count, src);
  Backend::template SendRecv<T>(sendbuf, send_count, dest,
                                recvbuf, recv_count, src, comm);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim SendRecv().
 *
 * @param[in,out] buffer Input and output buffer initially containing the
 * local data to send. Will be replaced with the received data.
 * @param[in] count Length of data to send and receive. \p buffer should
 * be `count` elements.
 * @param[in] dest Rank in comm to send to.
 * @param[in] src Rank in comm to receive from.
 * @param[in] comm Communicator to send/recv within.
 */
template <typename Backend, typename T>
void SendRecv(T* buffer, size_t count, int dest, int src,
              typename Backend::comm_type& comm) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(dest, comm);
  debug::check_rank<Backend>(src, comm);
  AL_CALI_MARK_SCOPE("aluminum:SendRecv");
  internal::trace::record_op<Backend, T>("sendrecv", comm, buffer, count,
                                         dest, src);
  Backend::template SendRecv<T>(buffer, count, dest, src, comm);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim SendRecv().
 *
 * @param[in] sendbuf Buffer containing the local data to send.
 * @param[in] send_count Length of \p sendbuf in elements of type `T`.
 * @param[in] dest Rank in comm to send to.
 * @param[out] recvbuf Buffer to receive the sent data.
 * @param[in] recv_count Length of \p recvbuf in elements of type `T`.
 * @param[in] src Rank in comm to receive from.
 * @param[in] comm Communicator to send/recv within.
 * @param[out] req Request object for the asynchronous operation.
 */
template <typename Backend, typename T>
void NonblockingSendRecv(const T* sendbuf, size_t send_count, int dest,
                         T* recvbuf, size_t recv_count, int src,
                         typename Backend::comm_type& comm,
                         typename Backend::req_type& req) {
  debug::check_buffer(sendbuf, send_count);
  debug::check_buffer(recvbuf, recv_count);
  debug::check_rank<Backend>(dest, comm);
  debug::check_rank<Backend>(src, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingSendRecv");
  internal::trace::record_op<Backend, T>("nonblocking-sendrecv", comm,
                                         sendbuf, send_count, dest,
                                         recvbuf, recv_count, src);
  Backend::template NonblockingSendRecv<T>(sendbuf, send_count, dest,
                                           recvbuf, recv_count, src,
                                           comm, req);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim SendRecv().
 *
 * @param[in,out] buffer Input and output buffer initially contaiuning the
 * local data to send. Will be replaced with the received data.
 * @param[in] count Length of data to send and receive. \p buffer should
 * be `count` elements.
 * @param[in] dest Rank in comm to send to.
 * @param[in] src Rank in comm to receive from.
 * @param[in] comm Communicator to send/recv within.
 * @param[out] req Request object for the asynchronous operation.
 */
template <typename Backend, typename T>
void NonblockingSendRecv(T* buffer, size_t count, int dest, int src,
                         typename Backend::comm_type& comm,
                         typename Backend::req_type& req) {
  debug::check_buffer(buffer, count);
  debug::check_rank<Backend>(dest, comm);
  debug::check_rank<Backend>(src, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingSendRecv");
  internal::trace::record_op<Backend, T>("nonblocking-sendrecv", comm,
                                         buffer, count, dest, src);
  Backend::template NonblockingSendRecv<T>(buffer, count, dest, src, comm, req);
}

/**
 * Perform an arbitrary sequence of Send() and Recv() operations.
 *
 * See \verbatim embed:rst:inline :ref:`MultiSendRecv <multisendrecv>`. \endverbatim
 *
 * @param[in] send_buffers Vector of buffers containing the local data to send.
 * @param[in] send_counts Vector of the lengths of each buffer in
 * \p send_buffers in elements of type `T`.
 * @param[in] dests Vector of the destination rank to send each buffer to.
 * @param[out] recv_buffers Vector of buffers to receive data in.
 * @param[in] recv_counts Vector of the lengths of each buffer in
 * \p recv_buffers in elements of type `T`.
 * @param[in] srcs Vector of the ranks to receive from.
 * @param[in] comm Communicator to send/recv within.
 */
template <typename Backend, typename T>
void MultiSendRecv(std::vector<const T*> send_buffers,
                   std::vector<size_t> send_counts, std::vector<int> dests,
                   std::vector<T*> recv_buffers,
                   std::vector<size_t> recv_counts, std::vector<int> srcs,
                   typename Backend::comm_type& comm) {
  debug::check_multisendrecv<Backend>(send_buffers, send_counts, dests,
                                      recv_buffers, recv_counts, srcs, comm);
  AL_CALI_MARK_SCOPE("aluminum:MultiSendRecv");
  internal::trace::record_op<Backend, T>("multisendrecv", comm,
                                         send_buffers, send_counts, dests,
                                         recv_buffers, recv_counts, srcs);
  Backend::template MultiSendRecv<T>(send_buffers, send_counts, dests,
                                     recv_buffers, recv_counts, srcs,
                                     comm);
}

/**
 * Perform an \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim MultiSendRecv().
 *
 * @param[in, out] buffers Vector of input and output buffers initially
 * containing the local data to send. Will be replaced with the received
 * data.
 * @param[in] counts Vector of the lengths of data to send and receive.
 * @param[in] dests Vector of the destination rank to send each buffer to.
 * @param[in] srcs Vector of the ranks to receive from.
 * @param[in] comm Communicator to send/recv within.
 */
template <typename Backend, typename T>
void MultiSendRecv(std::vector<T*> buffers, std::vector<size_t> counts,
                   std::vector<int> dests, std::vector<int> srcs,
                   typename Backend::comm_type& comm) {
  debug::check_inplace_multisendrecv<Backend>(buffers, counts, dests, srcs, comm);
  AL_CALI_MARK_SCOPE("aluminum:MultiSendRecv");
  internal::trace::record_op<Backend, T>("multisendrecv", comm,
                                         buffers, counts, dests, srcs);
  Backend::template MultiSendRecv<T>(buffers, counts, dests, srcs, comm);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim MultiSendRecv().
 *
 * @param[in] send_buffers Vector of buffers containing the local data to send.
 * @param[in] send_counts Vector of the lengths of each buffer in
 * \p send_buffers in elements of type `T`.
 * @param[in] dests Vector of the destination rank to send each buffer to.
 * @param[out] recv_buffers Vector of buffers to receive data in.
 * @param[in] recv_counts Vector of the lengths of each buffer in
 * \p recv_buffers in elements of type `T`.
 * @param[in] srcs Vector of the ranks to receive from.
 * @param[in] comm Communicator to send/recv within.
 * @param[out] req Request object for the asynchronous operation.
 */
template <typename Backend, typename T>
void NonblockingMultiSendRecv(
    std::vector<const T*> send_buffers, std::vector<size_t> send_counts,
    std::vector<int> dests, std::vector<T*> recv_buffers,
    std::vector<size_t> recv_counts, std::vector<int> srcs,
    typename Backend::comm_type& comm, typename Backend::req_type& req) {
  debug::check_multisendrecv<Backend>(send_buffers, send_counts, dests,
                                      recv_buffers, recv_counts, srcs, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingMultiSendRecv");
  internal::trace::record_op<Backend, T>("nonblocking-multisendrecv", comm,
                                         send_buffers, send_counts, dests,
                                         recv_buffers, recv_counts, srcs);
  Backend::template NonblockingMultiSendRecv<T>(send_buffers, send_counts, dests,
                                                recv_buffers, recv_counts, srcs,
                                                comm, req);
}

/**
 * Perform a \verbatim embed:rst:inline :ref:`non-blocking <comm-nonblocking>` \endverbatim
 * \verbatim embed:rst:inline :ref:`in-place <comm-inplace>` \endverbatim MultiSendRecv().
 *
 * @param[in, out] buffers Vector of input and output buffers initially
 * containing the local data to send. Will be replaced with the received
 * data.
 * @param[in] counts Vector of the lengths of data to send and receive.
 * @param[in] dests Vector of the destination rank to send each buffer to.
 * @param[in] srcs Vector of the ranks to receive from.
 * @param[in] comm Communicator to send/recv within.
 * @param[out] req Request object for the asynchronous operation.
 */
template <typename Backend, typename T>
void NonblockingMultiSendRecv(std::vector<T*> buffers,
                              std::vector<size_t> counts,
                              std::vector<int> dests, std::vector<int> srcs,
                              typename Backend::comm_type& comm,
                              typename Backend::req_type& req) {
  debug::check_inplace_multisendrecv<Backend>(buffers, counts, dests, srcs, comm);
  AL_CALI_MARK_SCOPE("aluminum:NonblockingMultiSendRecv");
  internal::trace::record_op<Backend, T>("nonblocking-multisendrecv", comm,
                                         buffers, counts, dests, srcs);
  Backend::template NonblockingMultiSendRecv<T>(buffers, counts, dests, srcs,
                                                comm, req);
}

/**
 * Return true if the asynchronous operation associated with \p req
 * has completed.
 *
 * This does not block. If the operation has completed, \p req will be
 * reset to `Backend::null_req`.
 *
 * See \verbatim embed:rst:inline :ref:`comm-nonblocking`. \endverbatim
 *
 * @param[in,out] req Request object for the asynchronous operation.
 */
template <typename Backend>
bool Test(typename Backend::req_type& req);

/**
 * Wait until the asynchronous operation associated with \p req has
 * has completed.
 *
 * This blocks the compute stream associated with the operation.
 *
 * \p req will be reset to `Backend::null_req` after the operation
 * completes.
 *
 * See \verbatim embed:rst:inline :ref:`comm-nonblocking`. \endverbatim
 *
 * @param[in,out] req Request object for the asynchronous operation.
 */
template <typename Backend>
void Wait(typename Backend::req_type& req);

/**
 * Register memory with a communicator.
 *
 * For certain backends and situations, this may improve the
 * performance of subsequent communication operations.
 *
 * @param[in] buf Buffer to register.
 * @param[in] count Length of \p buffer in elements of type `T`.
 * @param[in] comm Communicator to register the buffer with.
 */
template <typename Backend, typename T>
void RegisterMemory(T* buf, size_t count, typename Backend::comm_type& comm) {
  internal::trace::record_op<Backend, T>("register", comm, buf, count);
  Backend::template RegisterMemory<T>(buf, count, comm);
}

/**
 * Unregister memory with a communicator.
 *
 * The memory should have previously been registered with
 * RegisterMemory.
 *
 * @param[in] buf Buffer to unregister.
 * @param[in] comm Communicator to unregister the buffer from.
 */
template <typename Backend, typename T>
void UnregisterMemory(T* buf, typename Backend::comm_type& comm) {
  internal::trace::record_op<Backend, T>("unregister", comm, buf);
  Backend::template UnregisterMemory<T>(buf, comm);
}

namespace ext {

#ifdef AL_HAS_MPI_CUDA_RMA
/**
 * Attach a remote buffer to local memory space for RMA.
 * @param local_buf Local buffer attached by remote rank.
 * @param peer Rank in comm to attach buffers with.
 * @param comm Communicator to attach buffers.
 * @return Local address the remote buffer is attached at.
 */
template <typename Backend, typename T>
T *AttachRemoteBuffer(T *local_buf, int peer,
                      typename Backend::comm_type& comm) {
  return Backend::template AttachRemoteBuffer<T>(local_buf, peer, comm);
}

/**
 * Detach an attached buffer from local memory space.
 * @param remote_buf Buffer previously attached.
 * @param peer Rank in comm the buffer is attached with.
 * @param comm Communicator the buffer is attached with.
 */
template <typename Backend, typename T>
void DetachRemoteBuffer(T *remote_buf, int peer,
                        typename Backend::comm_type& comm) {
  Backend::template DetachRemoteBuffer<T>(remote_buf, peer, comm);
}

/**
 * Send a notification message.
 * @param peer Rank in comm to send a notification to.
 * @param comm Communicator to send a notification within.
 */
template <typename Backend>
void Notify(int peer, typename Backend::comm_type& comm) {
  Backend::Notify(peer, comm);
}

/**
 * Wait a notification message.
 * @param peer Rank in comm to wait a notification from.
 * @param comm Communicator to wait a notification within.
 */
template <typename Backend>
void Wait(int peer, typename Backend::comm_type& comm) {
  Backend::Wait(peer, comm);
}

/**
 * Exchange a notification message.
 * @param peer Rank in comm to exchange a notification with.
 * @param comm Communicator to exchange a notification within.
 */
template <typename Backend>
void Sync(int peer, typename Backend::comm_type& comm) {
  Backend::Sync(peer, comm);
}

/**
 * Exchange notification messages with multiple ranks.
 * @param peers Ranks in comm to exchange a notification with.
 * @param num_peers Number of ranks in peers.
 * @param comm Communicator to exchange notifications within.
 */
template <typename Backend>
void Sync(const int *peers, int num_peers,
          typename Backend::comm_type& comm) {
  Backend::Sync(peers, num_peers, comm);
}

/**
 * Put a point-to-point message.
 * @param srcbuf The data to put.
 * @param count Length of srcbuf.
 * @param dest Rank in comm to put to.
 * @param destbuf Buffer to put to.
 * @param comm Communicator to put within.
 */
template <typename Backend, typename T>
void Put(const T* srcbuf, int dest, T *destbuf,
         size_t count, typename Backend::comm_type& comm) {
  Backend::template Put<T>(srcbuf, dest, destbuf, count, comm);
}
#endif // AL_HAS_MPI_CUDA_RMA

} // namespace ext

}  // namespace Al

#include "aluminum/mpi_impl.hpp"

#ifdef AL_HAS_NCCL
#include "aluminum/nccl_impl.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "aluminum/mpi_cuda_impl.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "aluminum/ht_impl.hpp"
#endif

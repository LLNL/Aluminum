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
 * These are used to tune various algorithmic choices.
 * You should probably choose them based on benchmarks for your particular
 * configuration.
 */
#pragma once

/** Number of concurrent operations the progress engine will perform. */
#define AL_PE_NUM_CONCURRENT_OPS 4
/** Max number of streams the progress engine supports. */
#define AL_PE_NUM_STREAMS 64
/** Max number of pipeline stages the progress engine supports. */
#define AL_PE_NUM_PIPELINE_STAGES 2
/** Max number of entries in each stream's input queue. */
#define AL_PE_INPUT_QUEUE_SIZE 8192
/**
 * Whether to have a default stream entry for the progress engine
 * added automatically.
 *
 * This makes sense when using MPI, but not so when using the
 * host-transfer backend, which does not use the default stream.
 */
// #define AL_PE_ADD_DEFAULT_STREAM 1
/**
 * Whether to use a thread-local cache to map streams to input queues
 * for the progress engine.
 *
 * If you expect to have only a small number of streams, using a cache
 * is unlikely to help, since searching it will take as long as
 * searching the actual list.
 */
// #define AL_PE_STREAM_QUEUE_CACHE 1

/** Amount of sync object memory to preallocate in the pool. */
#define AL_SYNC_MEM_PREALLOC 1024

/**
 * Cache line size in bytes.
 *
 * On x86 this is usually 64. On POWER this is 128. On A64FX this is 256.
 */
#define AL_CACHE_LINE_SIZE 64

/**
 * Minimum size in bytes to avoid destructive interference.
 *
 * This is generally AL_CACHE_LINE_SIZE, except on x86, where it should
 * be twice the cache line size, because Intel processors can fetch
 * two adjacent cache lines (see Intel Optimization Manual, 3.7.3).
 */
#define AL_DESTRUCTIVE_INTERFERENCE_SIZE 128

/** Number of CUDA streams in the default stream pool. */
#define AL_CUDA_STREAM_POOL_SIZE 5

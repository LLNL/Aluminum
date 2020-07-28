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

/**
 * Number of concurrent operations the progress engine will perform.
 * This must be a positive number.
 */
#define AL_PE_NUM_CONCURRENT_OPS 4
/** Max number of streams the progress engine supports. */
#define AL_PE_NUM_STREAMS 64
/** Max number of pipeline stages the progress engine supports. */
#define AL_PE_NUM_PIPELINE_STAGES 2

/** Whether to protect memory pools with locks. */
#define AL_LOCK_MEMPOOL 1

/** Amount of sync object memory to preallocate in the pool. */
#define AL_SYNC_MEM_PREALLOC 1024

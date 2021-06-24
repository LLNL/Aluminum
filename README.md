# Aluminum

**Aluminum** provides a generic interface to high-performance communication libraries for both CPU and GPU platforms and GPU-friendly semantics.

If you use Aluminum, please cite our paper:
```
@inproceedings{dryden2018aluminum,
  title={Aluminum: An Asynchronous, {GPU}-Aware Communication Library Optimized for Large-Scale Training of Deep Neural Networks on {HPC} Systems},
  author={Dryden, Nikoli and Maruyama, Naoya and Moon, Tim and Benson, Tom and Yoo, Andy and Snir, Marc and Van Essen, Brian},
  booktitle={Proceedings of the Workshop on Machine Learning in HPC Environments (MLHPC)},
  year={2018}
}
```

## Features

* Support for blocking and non-blocking collective and point-to-point operations
* GPU-aware algorithms
* GPU-centric communication semantics
* Supported backends:
  * `MPI`: MPI and custom algorithms implemented on top of MPI
  * `NCCL`: Interface to Nvidia's [NCCL 2](https://developer.nvidia.com/nccl) library (including point-to-point operations and collectives built on them)
  * `HostTransfer`: Provides GPU support even when your MPI is not CUDA-aware
  * `MPI-CUDA`: Experimental intra-node RMA support
* Experimental support for AMD systems using HIP/ROCm and [RCCL](https://github.com/ROCmSoftwarePlatform/rccl)

### GPU-centric Communication

Aluminum aims to provide GPU-centric communication semantics with the `HostTransfer` backend: A communication operation should function "just like a CUDA kernel".
Aluminum supports associating a CUDA stream with a communicator.
All communication on the communicator will be with respect to the computation on that stream:
* The communication will proceed asynchronously and not block the initiating host thread.
* The communication will not begin until all outstanding operations on the stream at the time the communication operation was called have completed.
* No computation on the stream enqueued after the communication operation was will begin until the communication completes.

These semantics are comparable to those provided by NCCL.

### Non-blocking GPU Communication

Aluminum provides support for non-blocking GPU communication operations in its `NCCL` and `HostTransfer` backends.
Much like non-blocking MPI operations can be initiated by a thread, progress in the background, and be waited on for completion later, a CUDA stream can do the same thing.
Aluminum will manage the necessary synchronization and progress, and the communication will be performed on an internal CUDA stream.

## Getting started

Aluminum is also available via [Spack](https://spack.io/).

### Dependencies
For all builds:
* A compiler with at least C++14 support
* MPI (at least MPI 3.0)
* HWLOC (any recent version should work)
* CMake 3.17 or later

For GPU backends (`NCCL` and `HostTransfer`):
* CUDA (at least 9.0, for Nvidia GPUs) or HIP/ROCm (at least 3.6, for AMD GPUs)
* CUB (any recent version)

For the `NCCL`/`RCCL` backend:
* NCCL (for Nvidia GPUs) or RCCL (for AMD GPUs), at least version 2.7.0

### Building

Aluminum uses CMake. An out-of-source build is required.
A basic build can be done with:
```
mkdir build && cd build
cmake /path/to/aluminum/source
```

The supported communication backends are selected when you run `cmake`.
The `MPI` backend is always available. For other backends:
* `NCCL`: `-D ALUMINUM_ENABLE_NCCL=YES`
* `HostTransfer`: `-D ALUMINUM_ENABLE_HOST_TRANSFER=YES`
* `MPI-CUDA`: `-D ALUMINUM_ENABLE_MPI_CUDA=YES -D ALUMINUM_ENABLE_MPI_CUDA_RMA=YES`

To manually specify CUDA or ROCm support, use `-D ALUMINUM_ENABLE_CUDA=YES` or `-D ALUMINUM_ENABLE_ROCM=YES`.
If you specify a GPU communication backend, CUDA support will be assumed unless ROCm support is explicitly requested.

#### Other useful CMake flags

For specifying the MPI location, see the [CMake FindMPI documentation](https://cmake.org/cmake/help/latest/module/FindMPI.html).

To manually specify the a CUDA compiler, pass `-D CMAKE_CUDA_COMPILER=/path/to/nvcc`.

If HWLOC is installed in a nonstandard location, pass `-D HWLOC_DIR=/path/to/hwloc/prefix`.

If NCCL (or RCCL) is installed in a nonstandard location, pass `-D NCCL_DIR=/path/to/nccl/prefix`.

To specify an install directory, use the standard CMake flag: `-D CMAKE_INSTALL_PREFIX=/path/to/install/destination`.

##### Debug Builds

A standard debug build can be enabled by using `-D CMAKE_BUILD_TYPE=Debug`.
For additional debugging help, mostly intended for developers:
* Internal hang checking for the progress engine: `-D ALUMINUM_DEBUG_HANG_CHECK=YES`.
* Operation tracing: `-D ALUMINUM_ENABLE_TRACE=YES`.

### Example Build

For a "standard" Nvidia GPU system, you might use the following:
```
cmake \
-D ALUMINUM_ENABLE_NCCL=YES \
-D NCCL_DIR=/path/to/nccl \
-D ALUMINUM_ENABLE_HOST_TRANSFER=YES \
-D CMAKE_INSTALL_PREFIX=/path/to/install \
path/to/aluminum/source
```

## API Overview

The `MPI`, `NCCL`/`RCCL`, and `HostTransfer` backends support the following operations, including non-blocking and in-place (where meaingful) versions:
* Collectives:
  * Allgather
  * Vector allgather
  * Allreduce
  * Alltoall
  * Vector alltoall
  * Barrier
  * Broadcast
  * Gather
  * Vector gather
  * Reduce
  * ReduceScatter (equivalent to `MPI_Reduce_scatter_block`)
  * Vector ReduceScatter (equivalent to `MPI_Reduce_scatter`)
  * Scatter
  * Vector scatter
* Point-to-point:
  * Send
  * Recv
  * SendRecv

Full API documentation is coming soon...

## Tests and benchmarks

The `test` directory contains tests for every operation Aluminum supports.
The tests are only built when `-D ALUMINUM_ENABLE_TESTS=ON` is passed to CMake or a debug build is requested.

The main interface to the tests is `text_ops.exe`, which supports any combination of operation, backend, datatype, and so on that Aluminum supports.
For example, to test the `Alltoall` operation on the NCCL backend:
```
mpirun -n 128 ./test_ops.exe --op alltoall --backend nccl
```
Run it with `--help` for full details.

The `benchmark` directory contains benchmarks for all operations and can be run similarly using `benchmark_ops.exe`.
The benchmarks are only built when `-D ALUMINUM_ENABLE_BENCHMARKS=ON` is passed to CMake.

Note that building the benchmarks or tests can take a long time.

## Authors
* [Nikoli Dryden](https://github.com/ndryden)
* [Naoya Maruyama](https://github.com/naoyam)
* [Tom Benson](https://github.com/benson31)
* Andy Yoo

See also [contributors](https://github.com/ndryden/Aluminum/graphs/contributors).

## License

Aluminum is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

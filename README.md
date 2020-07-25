# Aluminum

**Aluminum** provides a generic interface to high-performance communication libraries, with a focus on allreduce algorithms. Blocking and non-blocking algorithms and GPU-aware algorithms are supported. Aluminum also contains custom implementations of select algorithms to optimize for certain situations.

## Features

* Blocking and non-blocking algorithms
* GPU-aware algorithms
* Implementations/interfaces:
  * `MPI`: MPI and custom algorithms implemented on top of MPI
  * `NCCL`: Interface to Nvidia's [NCCL 2](https://developer.nvidia.com/nccl) library
  * `MPI-CUDA`: Custom GPU-aware algorithms

## Getting started

### Prerequisites
* A compiler that supports at least C++11
* MPI (at least MPI 3.0)
* CUDA (at least 9.0, _optional if no GPU support is needed_)
* NCCL2 (_optional if no NCCL support is needed_)

### Building

CMake 3.9 or newer is required. An out-of-source build is required:
```
mkdir build && cd build
cmake <options> /path/to/aluminum/source
```

The required packages are `MPI` and `HWLOC`. `MPI`
uses the standard CMake package and can be manipulated in the
standard way. `HWLOC`, if installed in a nonstandard location, may
require `HWLOC_DIR` to be set to the appropriate installation prefix.

The `CUDA`-based backends assume `CUDA` is a first-class language in
CMake. An alternative `CUDA` compiler can be selected using
```
-DCMAKE_CUDA_COMPILER=/path/to/my/nvcc
```
If the `NCCL` backend is used, the `NCCL_DIR` variable may be
used to point CMake to a nonstandard installation prefix.

For the `NCCL` backend:
```
-DALUMINUM_ENABLE_NCCL=ON
```

For the `MPI-CUDA` backend:
```
-DALUMINUM_ENABLE_MPI_CUDA=ON
```

The `NCCL` and `MPI-CUDA` backends can be combined.

Here is a complete example:
```
CMAKE_PREFIX_PATH=/path/to/your/MPI:$CMAKE_PREFIX_PATH cmake -D ALUMINUM_ENABLE_NCCL=YES -D ALUMINUM_ENABLE_MPI_CUDA=YES -D NCCL_DIR=/path/to/NCCL ..
```

## Tests and benchmarks

The `test_correctness` binary will check the correctness of Aluminum's allreduce implementations. The usage is
```
test_correctness [Al backend: MPI, NCCL, MPI-CUDA]
```

For example, to test the `MPI` backend:
```
mpirun -n 128 ./test_correctness
```

To test the `NCCL` backend, instead:
```
mpirun -n 128 ./test_correctness NCCL
```

The `benchmark_allreduce` benchmark can be run similarly, and will report runtimes for different allreduce algorithms.

## API overview

Coming soon...

## Authors
* [Nikoli Dryden](https://github.com/ndryden)
* [Naoya Maruyama](https://github.com/naoyam)
* Andy Yoo
* Tom Benson

See also [contributors](https://github.com/ndryden/Aluminum/graphs/contributors).

## License

Aluminum is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

# Aluminum

**Aluminum** provides a generic interface to high-performance allreduce algorithms, with support for additional collective operations in progress. Blocking and non-blocking algorithms and GPU-aware algorithms are supported. Aluminum contains custom implementations and interfaces to existing optimized communication libraries, exposed through a generic API.

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
* CUDA (_optional if no GPU support is needed_)
* NCCL2 (_optional if no NCCL support is needed_)

### Building

This is currently ad-hoc. You may need to edit `Makefile` to adjust paths. Run `make` to build the `libAl` library and associated tests and benchmarks. This will build the `MPI` backend.

For the `NCCL` backend:
```
ENABLE_NCCL_CUDA=YES make
```

For the `MPI-CUDA` backend:
```
ENABLE_MPI_CUDA=YES make
```

The `NCCL` and `MPI-CUDA` backends can be combined.

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

See also [contributors](https://github.com/ndryden/Aluminum/graphs/contributors).

## License

Aluminum is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

![Al](al.svg) Aluminum
======================

**Aluminum** is a high-performance communication library for CPUs, GPUs, and other accelerator platforms.
It leverages existing libraries, such as MPI, NCCL, and RCCL, plus its own infrastructure, to deliver performance and accelerator-centric communication.

Aluminum is open-source and maintained by the Lawrence Livermore National Laboratory.
If you use Aluminum, please cite [our paper](https://ieeexplore.ieee.org/document/8638639):
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
* Accelerator-centric communication
* Supported communication backends:
  * `MPI`: Uses the Message Passing Interface and supports any hardware your underlying MPI library supports.
  * `NCCL`: Uses either Nvidia's [NCCL](https://developer.nvidia.com/nccl) library for Nvidia GPUs or AMD's [RCCL](https://github.com/ROCmSoftwarePlatform/rccl) library for AMD GPUs.
  * `HostTransfer`: Uses MPI plus the CUDA or HIP runtime to support Nvidia or AMD GPUs without specialized libraries.

## Getting Started

For full details, see the [Aluminum documentation](https://aluminum.readthedocs.io/).

For basic usage examples, see the [examples](examples).

### Building and Installation

Aluminum is available via [Spack](https://spack.io/) or can be installed manually from source.

Source builds need a recent CMake, C++ compiler (with support for C++17), MPI, and hwloc.
Accelerator backends need the appropriate runtime libraries.

A basic out-of-source build can be done with
```
mkdir build && cd build
cmake /path/to/Aluminum/source
```

For full details on building, configuration, testing, and benchmarking, see the [documentation](https://aluminum.readthedocs.io/en/latest/build.html).

## Authors

* [Nikoli Dryden](https://github.com/ndryden)
* [Naoya Maruyama](https://github.com/naoyam)
* [Tom Benson](https://github.com/benson31)
* Andy Yoo

See also [contributors](https://github.com/ndryden/Aluminum/graphs/contributors).

## License

Aluminum is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

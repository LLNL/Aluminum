Building Aluminum
=================

Aluminum is available via the `Spack package manager <https://spack.io>`_ as the ``aluminum`` package.

The rest of this document assumes you want to build Aluminum from source.
The following dependencies are required for all builds:

* CMake 3.21 or later
* A compiler with C++17 support
* MPI (with support for the MPI 3.0 standard or later)
* Hwloc (any recent version is sufficient)

The accelerator backends (``NCCL``, ``HostTransfer``) require:

* For Nvidia GPUs: CUDA 11.0 or later
* For AMD GPUs: HIP/ROCm 5.0 or later

The ``NCCL``/``RCCL`` backend additionally requires:

* For Nvidia GPUs: NCCL 2.14.0 or later
* For AMD GPUs: RCCL 2.14.0 or later (this is usually bundled with HIP/ROCm installs)

Aluminum uses CMake and an out-of-source build is required.
A basic build can be done with:

.. code-block:: bash

   mkdir build && cd build
   cmake /path/to/Aluminum/source

Communication backends and other features are selected when you run CMake (see :ref:`configopts`).
There are some standard CMake flags which may be useful:

* See the `CMake FindMPI documentation <https://cmake.org/cmake/help/latest/module/FindMPI.html>`_ for information on specifying the MPI location if it is not found automatically.
* To manually specify a CUDA compiler, pass ``-D CMAKE_CUDA_COMPILER=/path/to/nvcc``.
* To manually specify the hwloc location, pass ``-D HWLOC_DIR=/path/to/hwloc/prefix``.
* To manually specify the NCCL or RCCL install location, pass ``-D NCCL_DIR=/path/to/nccl/prefix``.
* To request a debug build, pass ``-D CMAKE_BUILD_TYPE=Debug``.
* To specify an install directory, pass ``-D CMAKE_INSTALL_PREFIX=/path/to/install/dir``.
* To export compile commands (for developers), pass ``-D CMAKE_EXPORT_COMPILE_COMMANDS=YES``.

.. _configopts:

Configuration Options
---------------------

To check what features Aluminum was built with, run the ``al_info`` binary.

Communication backends
^^^^^^^^^^^^^^^^^^^^^^

These options select which of Aluminum's communication backends are enabled.
The MPI backend is always enabled and cannot be disabled.
For accelerator backends, CUDA support is automatically assumed unless support for another accelerator is explicitly requested (see :ref:`accelsupport`).
You can enable any combination of backends.

* ``NCCL`` (using Nvidia's NCCL library on Nvidia GPUs or AMD's RCCL library on AMD GPUs): ``-D ALUMINUM_ENABLE_NCCL=YES``
* ``HostTransfer`` (using MPI plus the CUDA or HIP runtime to support GPUs): ``-D ALUMINUM_ENABLE_HOST_TRANSFER=YES``

.. _accelsupport:

Accelerator support
^^^^^^^^^^^^^^^^^^^

These options select which accelerators Aluminum will support, if any.
Only one accelerator type can be enabled at a time.
(Note CPUs are always supported.)
If you request an accelerator communication backend (``NCCL`` or ``HostTransfer``), CUDA support will be automatically enabled unless support for a different accelerator is requested.

* CUDA support (for Nvidia GPUs): ``-D ALUMINUM_ENABLE_CUDA=YES``
* HIP/ROCm support (for AMD GPUs): ``-D ALUMINUM_ENABLE_ROCM=YES``

Hence, if you want to use RCCL on AMD GPUs, you need to pass ``-D ALUMINUM_ENABLE_NCCL=YES -D ALUMINUM_ENABLE_ROCM=YES``.

Testing and bechmarking
^^^^^^^^^^^^^^^^^^^^^^^

These options enable Aluminum's :ref:`tests <testing>` or :ref:`benchmarks <benchmarking>`.
These are disabled by default, as they can take a long time to build.

* Enable tests: ``-D ALUMINUM_ENABLE_TESTS=YES``
* Enable benchmarks: ``-D ALUMINUM_ENABLE_BENCHMARKS=YES``

Thread safety
^^^^^^^^^^^^^

Aluminum supports configurable thread-safety levels; see :ref:`thread-safety` for details.

* Enable support for concurrent use of Aluminim: ``-D ALUMINUM_ENABLE_THREAD_MULTIPLE=YES``
* Serialize MPI calls: ``-D ALUMINUM_MPI_SERIALIZE=YES``

  This is designed for cases where your MPI library requires all MPI calls to come from a single thread.
  Aluminum will funnel all its MPI calls to its internal progress engine in this configuration.

Debugging and profiling
^^^^^^^^^^^^^^^^^^^^^^^

Aluminum supports several configurations to aid debugging and profiling.
These are primarily indended for developers.

* Internal hang checking for the progress engine: ``-D ALUMINUM_ENABLE_HANG_CHECK=YES``
* Trace all Aluminum calls: ``-D ALUMINUM_ENABLE_TRACE=YES``
* Enable profiling annotations via nvprof/NVTX: ``-D ALUMINUM_ENABLE_NVPROF=YES``
* Enable profiling annotations via rocprof/roctx: ``-D ALUMINUM_ENABLE_ROCTRACER=YES``

Miscellaneous performance knobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A number of internal Aluminum parameters are configurable.
These generally have reasonable default values, but can be tuned to improve performance for specific systems and/or workloads.
Since these may change without notice, these are not documented here.
Rather, see the CMake help or the ``cmake/tuning_params.hpp.in`` file.

.. _testing:

Testing
-------

Aluminum has an interface for running any supported operation (with any backend or datatype) and comparing its result with what MPI produces.
This is primarily driven by the ``test_ops`` binary, and you should see its ``--help`` output for full details.

*Note*: In the case of the ``half`` (FP16) and ``bfloat16`` datatypes and reduction operations, the ground-truth MPI checks will use single-precison (FP32), as MPI does not currently support ``half`` or ``bfloat16``.
This can result in spurious failures.

Additionally, there is a Python wrapper script, ``run_tests.py``, which can be used to automate sweeping over many configurations.
Again, see its ``--help`` output for full details.

.. _benchmarking:

Benchmarking
------------

Aluminum also has a similar interface for benchmarking any supported operation.
It also supports standard benchmarking conveniences, including warmups and using accelerator-side timers when timing such operations (e.g., using CUDA events).
This functions similarly to testing, and is driven by the ``benchmark_ops`` binary.
There is also a Python wrapper script, ``run_benchmarks.py``, for sweeping benchmark configurations, and ``plot_benchmarks.py`` to generate plots.

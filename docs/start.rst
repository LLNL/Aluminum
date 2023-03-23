Getting Started with Aluminum
=============================

Once you have :doc:`built <build>` Aluminum, you probably want to use it.
Aluminum is in many respects quite similar to MPI, so if you have experience with that, things should be familiar.

This is a simple "Hello, world" program that shows key aspects of Aluminum:

.. code-block:: c++

   #include <Al.hpp>
   #include <iostream>

   int main(int argc, char** argv) {
     // Initialize Aluminum.
     // Must be called before any other Aluminum calls.
     Al::Initialize(argc, argv);

     // Create a communicator with all processes.
     typename Al::MPIBackend::comm_type comm;

     // Each process prints its rank and the communicator size.
     std::cout << "Hello, world, from rank "
               << comm.rank() << " of "
               << comm.size() << std::endl;

     // Do a simple (in-place) allreduce.
     int rank = comm.rank();
     Al::Allreduce<MPIBackend>(&rank, 1, Al::ReductionOperator::sum, comm);
     std::cout << "The sum of ranks is "
               << rank << std::endl;

     // Clean up Aluminum.
     Al::Finalize();

     return 0;
   }

For additional examples and more detail (including accelerator backends), see the `Aluminum examples <https://github.com/LLNL/Aluminum/tree/master/examples>`_.

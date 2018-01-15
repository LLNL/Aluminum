/**
 * These are used to tune various algorithmic choices.
 * You should probably choose them based on benchmarks for your particular
 * configuration.
 */
#pragma once

/** Whether to use OpenMP for reduction operators. */
#define ALLREDUCE_MPI_USE_OPENMP 0
/** Use multiple threads for sum reductions this size or larger. */
#define ALLREDUCE_MPI_MULTITHREAD_SUM_THRESH 262144
/** Use multiple threads for prod reductions this size or larger. */
#define ALLREDUCE_MPI_MULTITHREAD_PROD_THRESH 262144
/** Use multiple threads for min/max reductions this size or larger. */
// Note: May need to re-benchmark this to deal with branch prediction.
#define ALLREDUCE_MPI_MULTITHREAD_MINMAX_THRESH 262144

/**
 * Whether the progress engine just polls constantly or sleeps when there is no
 * work to do.
 * Sleeping will reduce unneeded CPU usage (probably helps if you're also doing
 * compute on the same core) but probably increases latency a little bit.
 */
#define ALLREDUCE_PE_SLEEPS 1

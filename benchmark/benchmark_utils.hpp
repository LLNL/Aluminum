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

#pragma once

#include "Al.hpp"
#include <unordered_map>
#include <fstream>
#include "test_utils.hpp"


/** Handle timing for a particular backend. */
template <typename Backend>
struct Timer {
  /** Start the timer. */
  void start_timer(typename Backend::comm_type& comm);
  /** End the timer and return the elapsed time. */
  double end_timer(typename Backend::comm_type& comm);
};


/** Summary results for a benchmark. */
struct SummaryStats {
  SummaryStats() {};
  /**
   * Compute summary statistics for values in v.
   *
   * Note we do not attempt to compute stats in a numerically stable manner.
   *
   * Note this will modify v by doing a partial sort to compute the median.
   */
  SummaryStats(std::vector<double>& v) {
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    mean = sum / v.size();
    if (v.size() > 1) {
      double sqsum = 0.0;
      for (const auto& x : v) {
        sqsum += (x - mean) * (x - mean);
      }
      stdev = std::sqrt(1.0 / (v.size() - 1) * sqsum);
      std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
      // This is not correct for even-length vectors, but to quote
      // Numerical Recipes: "... formalists be damned".
      median = v[v.size() / 2];
      auto minmax = std::minmax_element(v.begin(), v.end());
      min = *minmax.first;
      max = *minmax.second;
    }
  }
  double mean = std::numeric_limits<double>::quiet_NaN();
  double stdev = std::numeric_limits<double>::quiet_NaN();
  double median = std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
};

inline std::ostream& operator<<(std::ostream& os, const SummaryStats& summary) {
  os << summary.mean << " "
     << summary.median << " "
     << summary.stdev << " "
     << summary.min << " "
     << summary.max;
  return os;
}


/** Collect and output performance results. */
template <AlOperation Op, typename Backend, typename T>
class OpProfile {
public:
  OpProfile(typename Backend::comm_type& comm_,
            OpOptions<Backend>& options_) :
    comm(comm_), options(options_) {}

  void add_result(size_t size, double time) {
    if (times.count(size) == 0) {
      times[size] = {};
    }
    times[size].push_back(time);
  }

  /**
   * Return summary stats over all ranks (-1) or a specific rank.
   *
   * Summary is returned only on rank 0.
   */
  std::unordered_map<size_t, SummaryStats> get_summary_stats(int rank = -1) {
    auto gathered_times = gather_results_to_root();
    if (comm.rank() == 0) {
      std::unordered_map<size_t, SummaryStats> summaries;
      for (auto&& p : gathered_times) {
        size_t size = p.first;
        std::vector<double>& ts = p.second;
        size_t times_per_rank = ts.size() / comm.size();
        if (rank > 0) {
          std::vector<double> rank_ts(ts.begin() + rank*times_per_rank,
                                      ts.begin() + (rank+1)*times_per_rank);
          summaries[size] = SummaryStats(rank_ts);
        } else {
          summaries[size] = SummaryStats(ts);
        }
      }
      return summaries;
    } else {
      return {};
    }
  }

  void print_results() {
    write_results(std::cout);
  }

  void save_results(std::string filename) {
    std::ofstream f(filename);
    if (f.fail()) {
      std::cerr << "Error opening " << filename << std::endl;
      std::abort();
    }
    write_results(f);
  }

  template <AlOperation Op2 = Op,
            std::enable_if_t<IsCollectiveOp<Op2>::value
                             && IsOpSupported<Op2, Backend>::value, bool> = true>
  void write_results(std::ostream& os) {
    // Write times.
    auto gathered_times = gather_results_to_root();
    if (comm.rank() == 0) {
      // Header.
      os << "Backend Type Operation Algo NonBlocking InPlace"
         << " Root CommSize Size CommRank Time\n";
      AlgoAccessor<Op, Backend> getter;
      const std::string common_start =
        std::string(AlBackendName<Backend>) + " "
        + std::string(typeid(T).name()) + " "
        + std::string(AlOperationName<Op>) + " "
        + Al::algorithm_name(getter.get(options.algos)) + " "
        + (options.nonblocking ? "1" : "0") + " "
        + (options.inplace ? "1" : "0") + " "
        + std::to_string(options.root) + " "
        + std::to_string(comm.size()) + " ";
      for (auto&& p : gathered_times) {
        size_t size = p.first;
        std::vector<double>& ts = p.second;
        size_t times_per_rank = ts.size() / comm.size();
        for (int rank = 0; rank < comm.size(); ++rank) {
          for (size_t i = rank * times_per_rank;
               i < (rank + 1) * times_per_rank;
               ++i) {
            os << common_start
               << size << " "
               << rank << " "
               << ts[i] << "\n";
          }
        }
      }
    }
  }

  template <AlOperation Op2 = Op,
            std::enable_if_t<IsPt2PtOp<Op2>::value
                             && IsOpSupported<Op2, Backend>::value, bool> = true>
  void write_results(std::ostream& os) {
    // Write times.
    auto gathered_times = gather_results_to_root();
    if (comm.rank() == 0) {
      // Header.
      os << "Backend Type Operation CommSize Size CommRank Time\n";
      const std::string common_start =
        std::string(AlBackendName<Backend>) + " "
        + std::string(typeid(T).name()) + " "
        + std::string(AlOperationName<Op>) + " "
        + std::to_string(comm.size()) + " ";
      for (auto&& p : gathered_times) {
        size_t size = p.first;
        std::vector<double>& ts = p.second;
        size_t times_per_rank = ts.size() / comm.size();
        for (int rank = 0; rank < comm.size(); ++rank) {
          for (size_t i = rank * times_per_rank;
               i < (rank + 1) * times_per_rank;
               ++i) {
            os << common_start
               << size << " "
               << rank << " "
               << ts[i] << "\n";
          }
        }
      }
    }
  }

  template <AlOperation Op2 = Op,
            std::enable_if_t<(!IsCollectiveOp<Op2>::value && !IsPt2PtOp<Op2>::value)
                             || !IsOpSupported<Op2, Backend>::value, bool> = true>
  void write_results(std::ostream& os) {
    os << "Unsupported operation" << std::endl;
  }

private:
  typename Backend::comm_type& comm;
  OpOptions<Backend>& options;

  std::unordered_map<size_t, std::vector<double>> times;

  std::unordered_map<size_t, std::vector<double>> gather_results_to_root() {
    if (comm.rank() == 0) {
      // Assumes all ranks have run the exact same sequence of tests.
      std::unordered_map<size_t, std::vector<double>> gathered_times;
      for (auto&& i : times) {
        size_t size = i.first;
        std::vector<double>& t = i.second;
        gathered_times.emplace(
          std::make_pair(size, std::vector<double>(t.size()*comm.size())));
        MPI_Gather(t.data(), t.size(), MPI_DOUBLE,
                   gathered_times[size].data(), t.size(), MPI_DOUBLE,
                   0, comm.get_comm());
      }
      return gathered_times;
    } else {
      for (auto&& i : times) {
        MPI_Gather(i.second.data(), i.second.size(), MPI_DOUBLE,
                   nullptr, 0, MPI_DOUBLE, 0, comm.get_comm());
      }
      return {};
    }
  }
};


#include "benchmark_utils_mpi.hpp"
#ifdef AL_HAS_NCCL
#include "benchmark_utils_nccl.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "benchmark_utils_ht.hpp"
#endif

#include <iostream>
#include <functional>
#include <algorithm>
#include <mpi.h>
#include "test_utils.hpp"

const size_t max_size = 1<<30;
const size_t num_trials = 100;

template <typename T>
void sum_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] += src[i];
  }
}
template <typename T>
void mt_sum_reduction(const T* src, T* dest, size_t count) {
  if (count >= 32768) {
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] += src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] += src[i];
    }
  }
}

template <typename T>
void prod_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] *= src[i];
  }
}
template <typename T>
void mt_prod_reduction(const T* src, T* dest, size_t count) {
  if (count >= 8192) {
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] *= src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] *= src[i];
    }
  }
}

template <typename T>
void min_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = std::min(dest[i], src[i]);
  }
}
template <typename T>
void mt_min_reduction(const T* src, T* dest, size_t count) {
  if (count >= 8192) {
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::min(dest[i], src[i]);
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::min(dest[i], src[i]);
    }
  }
}

template <typename T>
void max_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = std::max(dest[i], src[i]);
  }
}
template <typename T>
void mt_max_reduction(const T* src, T* dest, size_t count) {
  if (count >= 8192) {
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::max(dest[i], src[i]);
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::max(dest[i], src[i]);
    }
  }
}

void time_reduction(
  size_t count,
  std::function<void(const float*, float*, size_t)> single,
  std::function<void(const float*, float*, size_t)> multi,
  std::string name) {
  std::vector<double> times, multi_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    // Note: May need to clear cache between trials.
    std::vector<float> data = gen_data(count);
    std::vector<float> multi_data(data);
    std::vector<float> out(data), multi_out(data);
    double start = get_time();
    single(data.data(), out.data(), data.size());
    times.push_back(get_time() - start);
    start = get_time();
    multi(multi_data.data(), multi_out.data(), data.size());
    multi_times.push_back(get_time() - start);
  }
  times.erase(times.begin());
  multi_times.erase(multi_times.begin());
  std::cout << "size=" << count << " algo=" << name << " single ";
  print_stats(times);
  std::cout << "size=" << count << " algo=" << name << " multi ";
  print_stats(multi_times);
}

int main(int, char*[]) {
  for (size_t size = 1; size <= max_size; size *= 2) {
    time_reduction(size, sum_reduction<float>, mt_sum_reduction<float>, "sum");
    time_reduction(size, prod_reduction<float>, mt_prod_reduction<float>, "prod");
    time_reduction(size, min_reduction<float>, mt_min_reduction<float>, "min");
    time_reduction(size, max_reduction<float>, mt_max_reduction<float>, "max");
  }
  return 0;
}

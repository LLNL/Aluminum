#pragma once

#include "progress.hpp"

namespace Al {

/**
 * Internal implementations.
 * Generic code for all collective implementations is in here.
 * Implementation-specific code is in separate namespaces inside internal.
 */
namespace internal {

// Would be nice to replace this with a C++14 variable template...
/** Indicator that an in-place allreduce is requested. */
template <typename T>
inline T* IN_PLACE() { return (T*) (-1); }

}  // namespace internal
}  // namespace Al

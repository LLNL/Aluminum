#pragma once

#include <exception>
#include <string>

namespace Al {

/**
 * Base allreduce exception class.
 */
class allreduce_exception : public std::exception {
 public:
  allreduce_exception(const std::string m, const std::string f, const int l) :
    msg(m), file(f), line(l) {
    err = file + ":" + std::to_string(line) + " - " + msg;
  }
  const char* what() const noexcept override {
    return err.c_str();
  }
private:
  /** Exception message. */
  const std::string msg;
  /** File exception occurred in. */
  const std::string file;
  /** Line exception occurred at. */
  const int line;
  /** Constructed error message. */
  std::string err;
};
#define throw_allreduce_exception(s) throw allreduce_exception(s, __FILE__, __LINE__)

/** Predefined reduction operations. */
enum class ReductionOperator {
  sum, prod, min, max
};

} // namespace Al

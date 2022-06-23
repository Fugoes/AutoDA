#ifndef AUTODA_PRELUDE_HPP
#define AUTODA_PRELUDE_HPP

extern "C" {
#include <tensorflow/c/c_api.h>
#include <unistd.h>
};

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>

#include <iostream>
#include <ostream>
#include <random>
#include <memory>
#include <vector>
#include <tuple>
#include <stack>
#include <exception>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "dynarray.hpp"
#include <experimental/source_location>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <optional>

#include <Eigen/Dense>

#define autoda_norm(x) (x).matrix().norm()
#define autoda_clip(x) (x).cwiseMax(0.0).cwiseMin(1.0)
#define autoda_fill_randn(x) \
  do { \
    size_t __rows = (x).rows(); \
    for (size_t __i = 0; __i < __rows; __i++) { \
      (x)(__i) = ::autoda::_normal_dist(::autoda::gen); \
    } \
  } while (0)
#define autoda_fill_randu(x) \
  do { \
    size_t __rows = (x).rows(); \
    for (size_t __i = 0; __i < __rows; __i++) { \
      (x)(__i) = ::autoda::_uniform_dist(::autoda::gen); \
    } \
  } while (0)

namespace autoda {

inline void bug(
    const std::experimental::source_location &loc = std::experimental::source_location::current()) {
  std::ostringstream os{};
  os << loc.file_name() << ":" << loc.line() << ":" << loc.function_name();
  throw std::logic_error(os.str());
}

inline void check(bool flag,
                  const std::experimental::source_location &loc = std::experimental::source_location::current()) {
  if (!flag) {
    bug(loc);
  }
}

inline void check(TF_Status *status,
                  const std::experimental::source_location &loc = std::experimental::source_location::current()) {
  if (TF_GetCode(status) != TF_OK) {
    std::cerr << TF_Message(status);
    bug(loc);
  }
}

struct Stopwatch {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_{};

  Stopwatch() {}

  inline void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  inline size_t stop() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
  }
};

template<typename S>
struct TFTensor;

template<>
struct TFTensor<float> {
  TF_Tensor *raw_{nullptr};
  float *data_{nullptr};

  TFTensor() = default;
  TFTensor(const fstd::dynarray<int64_t> &dims) {
    check(dims.size() >= 1);
    int num_dims = dims.size();
    size_t len = sizeof(float);
    for (auto dim : dims) { len *= dim; }
    raw_ = TF_AllocateTensor(TF_FLOAT, dims.data(), num_dims, len);
    check(raw_ != nullptr);
    data_ = (float *) TF_TensorData(raw_);
  }
  TFTensor(TF_Tensor *raw) {
    raw_ = raw;
    check(TF_TensorType(raw_) == TF_FLOAT);
    data_ = (float *) TF_TensorData(raw_);
  }
  ~TFTensor() {
    if (raw_) { TF_DeleteTensor(raw_); }
  }
  TFTensor(const TFTensor &) = delete;
  TFTensor(TFTensor &&that) noexcept {
    raw_ = that.raw_;
    data_ = that.data_;
    that.raw_ = nullptr;
    that.data_ = nullptr;
  }
  inline void swap(TFTensor<float> &that) noexcept {
    auto this_raw = raw_;
    auto this_data = data_;
    raw_ = that.raw_;
    data_ = that.data_;
    that.raw_ = this_raw;
    that.data_ = this_data;
  }

  inline float *get_data() noexcept {
    return data_;
  }
};

template<>
struct TFTensor<unsigned char> {
  TF_Tensor *raw_{nullptr};
  unsigned char *data_{nullptr};

  TFTensor() = default;
  TFTensor(const fstd::dynarray<int64_t> &dims) {
    check(dims.size() >= 1);
    int num_dims = dims.size();
    size_t len = sizeof(unsigned char);
    for (auto dim : dims) { len *= dim; }
    raw_ = TF_AllocateTensor(TF_FLOAT, dims.data(), num_dims, len);
    check(raw_ != nullptr);
    data_ = (unsigned char *) TF_TensorData(raw_);
  }
  TFTensor(TF_Tensor *raw) {
    raw_ = raw;
    check(TF_TensorType(raw_) == TF_BOOL);
    data_ = (unsigned char *) TF_TensorData(raw_);
  }
  ~TFTensor() {
    if (raw_) { TF_DeleteTensor(raw_); }
  }
  TFTensor(const TFTensor &) = delete;
  TFTensor(TFTensor &&that) noexcept {
    raw_ = that.raw_;
    data_ = that.data_;
    that.raw_ = nullptr;
    that.data_ = nullptr;
  }
  inline void swap(TFTensor<unsigned char> &that) noexcept {
    auto this_raw = raw_;
    auto this_data = data_;
    raw_ = that.raw_;
    data_ = that.data_;
    that.raw_ = this_raw;
    that.data_ = this_data;
  }

  inline unsigned char *get_data() noexcept {
    return data_;
  }
};

template<typename ... Args>
inline void logi(const std::string msg, Args ... args) {
  auto fmt = "[I] " + std::string(msg) + "\n";
  fprintf(stdout, fmt.c_str(), args ...);
}

inline void logi(const std::string msg) {
  auto fmt = "[I] " + std::string(msg) + "\n";
  std::cout << fmt;
}

}

#endif //AUTODA_PRELUDE_HPP

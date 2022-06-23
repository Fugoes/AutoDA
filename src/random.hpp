#ifndef AUTODA_RANDOM_HPP
#define AUTODA_RANDOM_HPP

#include "prelude.hpp"

namespace autoda {

thread_local static std::random_device _rd{};
thread_local static std::seed_seq _seed{_rd(), _rd(), _rd(), _rd(), _rd(), _rd(), _rd(), _rd()};
thread_local static std::mt19937 gen(_seed);
thread_local static std::normal_distribution<float> _normal_dist(0, 1);
thread_local static std::uniform_real_distribution<float> _uniform_dist(0, 1);

template<typename T>
inline T random_select_from(const std::vector<T> &xs) {
  std::uniform_int_distribution<size_t> dist(0, xs.size() - 1);
  return xs[dist(gen)];
}

}

#endif //AUTODA_RANDOM_HPP

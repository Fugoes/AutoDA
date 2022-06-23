#ifndef AUTODA_CONFIG_HPP
#define AUTODA_CONFIG_HPP

#include "prelude.hpp"

namespace autoda {

struct TestConfig {
  std::string dir_{};
};

TestConfig config_from_args(int argc, char **argv);

struct AblationConfig {
  std::string dir_{};
  unsigned threads_{};
  unsigned gen_threads_{};
  unsigned class_0_{};
  unsigned class_1_{};
  unsigned cpu_batch_size_{};
  unsigned gpu_batch_size_{};
  size_t count_{};
  std::string method_{};
  std::string output_{};

  static AblationConfig from_args(int argc, char **argv);
};

struct LRAConfig {
  std::string dir_{};
  unsigned threads_{};
  unsigned gen_threads_{};
  unsigned class_0_{};
  unsigned class_1_{};
  unsigned cpu_batch_size_{};
  unsigned gpu_batch_size_{};
  std::string output_{};
  size_t max_queries_{};

  static LRAConfig from_args(int argc, char **argv);
};

}

#endif //AUTODA_CONFIG_HPP

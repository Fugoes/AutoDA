#ifndef AUTODA_OUTPUT_HPP
#define AUTODA_OUTPUT_HPP

namespace autoda {

struct output {
  ~output() {
    instance().os_.close();
  }

  static void initialize(const std::string &filename) {
    instance().os_.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);
  }

  static void append(std::stringstream &ss) {
    std::lock_guard<decltype(mutex_)> lock(instance().mutex_);
    instance().os_ << ss.str();
    instance().os_ << "##########\n";
  }

  static void flush() {
    std::lock_guard<decltype(mutex_)> lock(instance().mutex_);
    instance().os_.flush();
  }

 private:
  std::mutex mutex_{};
  std::ofstream os_{};

  static output &instance() {
    static output singleton{};
    return singleton;
  }
};

}

#endif //AUTODA_OUTPUT_HPP

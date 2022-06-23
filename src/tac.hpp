#ifndef AUTODA_TAC_HPP
#define AUTODA_TAC_HPP

#include "prelude.hpp"
#include "ops.hpp"
#include "random.hpp"

namespace autoda {

// All scalar storage and vector storage are put at the beginning.

struct TacVM {
  typedef unsigned (*DispatchFnPtr)(TacVM &vm, const std::vector<uint8_t> &program, size_t pc);

  static DispatchFnPtr dispatch_table_[];

 public:
  Eigen::Array<float, Eigen::Dynamic, 1> S_;
  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> V_;

  TacVM(size_t dim, uint8_t S_count, uint8_t V_count);
  ~TacVM() = default;

  void display(std::ostream &os);
};

struct TacProgram {
  std::vector<uint8_t> tac_{};
  uint8_t S_slots_{};
  uint8_t V_slots_{};

  void emit(uint8_t op, uint8_t _1, uint8_t _0) {
    tac_.push_back(op);
    tac_.push_back(_1);
    tac_.push_back(_0);
  }

  void emit(uint8_t op, uint8_t _1, uint8_t _2, uint8_t _0) {
    tac_.push_back(op);
    tac_.push_back(_1);
    tac_.push_back(_2);
    tac_.push_back(_0);
  }

  void display(std::ostream &os) const;
  void execute(TacVM &vm) const;
};

}

#endif //AUTODA_TAC_HPP

#include "tac.hpp"

#include <cmath>

namespace {

unsigned dispatch_0_SS_S_ADD(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.S_(_0) = vm.S_(_1) + vm.S_(_2);
  return 4;
}

unsigned dispatch_1_SS_S_SUB(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.S_(_0) = vm.S_(_1) - vm.S_(_2);
  return 4;
}

unsigned dispatch_2_SS_S_MUL(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.S_(_0) = vm.S_(_1) * vm.S_(_2);
  return 4;
}

unsigned dispatch_3_SS_S_DIV(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.S_(_0) = vm.S_(_1) / vm.S_(_2);
  return 4;
}

unsigned dispatch_4_VV_V_ADD(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) + vm.V_.col(_2);
  return 4;
}

unsigned dispatch_5_VV_V_SUB(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) - vm.V_.col(_2);
  return 4;
}

unsigned dispatch_6_VV_V_MUL(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) * vm.V_.col(_2);
  return 4;
}

unsigned dispatch_7_VV_V_DIV(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) / vm.V_.col(_2);
  return 4;
}

unsigned dispatch_8_VS_V_ADD(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) + vm.S_(_2);
  return 4;
}

unsigned dispatch_9_VS_V_SUB(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) - vm.S_(_2);
  return 4;
}

unsigned dispatch_10_SV_V_SUB(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.S_(_1) - vm.V_.col(_2);
  return 4;
}

unsigned dispatch_11_VS_V_MUL(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) * vm.S_(_2);
  return 4;
}

unsigned dispatch_12_VS_V_DIV(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.V_.col(_1) / vm.S_(_2);
  return 4;
}

unsigned dispatch_13_SV_V_DIV(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.V_.col(_0) = vm.S_(_1) / vm.V_.col(_2);
  return 4;
}

unsigned dispatch_14_VV_S_DOT(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  vm.S_(_0) = vm.V_.col(_1).matrix().dot(vm.V_.col(_2).matrix());
  return 4;
}

unsigned dispatch_15_SS_V_NORMAL(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  autoda_fill_randn(vm.V_.col(_0));
  auto mean = vm.S_(_1);
  auto stddev = vm.S_(_2);
  vm.V_.col(_0) = stddev * vm.V_.col(_0) + mean;
  return 4;
}

unsigned dispatch_16_SS_V_UNIFORM(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _2 = program[pc + 2], _0 = program[pc + 3];
  float a = vm.S_(_1), b = vm.S_(_2);
  autoda_fill_randu(vm.V_.col(_0));
  vm.V_.col(_0) = (b - a) * vm.V_.col(_0) + a;
  return 4;
}

unsigned dispatch_17_S_S_SQUARE(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.S_(_0) = vm.S_(_1) * vm.S_(_1);
  return 3;
}

unsigned dispatch_18_V_V_SQUARE(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.V_.col(_0) = vm.V_.col(_1).square();
  return 3;
}

unsigned dispatch_19_V_S_NORM(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.S_(_0) = vm.V_.col(_1).matrix().norm();
  return 3;
}

unsigned dispatch_20_V_S_SUM(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.S_(_0) = vm.V_.col(_1).sum();
  return 3;
}

unsigned dispatch_21_V_V_SQRT_ABS(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.V_.col(_0) = vm.V_.col(_1).abs().sqrt();
  return 3;
}

unsigned dispatch_22_S_S_SQRT_ABS(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.S_(_0) = sqrtf(fabsf(vm.S_(_1)));
  return 3;
}

unsigned dispatch_23_S_S_RELU(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.S_(_0) = vm.S_(_1) >= 0.0 ? vm.S_(_1) : 0.0;
  return 3;
}

unsigned dispatch_24_V_V_RELU(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.V_.col(_0) = vm.V_.col(_1).cwiseMax(0.0);
  return 3;
}

unsigned dispatch_25_S_S_COPY(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.S_(_0) = vm.S_(_1);
  return 3;
}

unsigned dispatch_26_V_V_COPY(autoda::TacVM &vm, const std::vector<uint8_t> &program, size_t pc) {
  uint8_t _1 = program[pc + 1], _0 = program[pc + 2];
  vm.V_.col(_0) = vm.V_.col(_1);
  return 3;
}

}

namespace autoda {

TacVM::DispatchFnPtr TacVM::dispatch_table_[] = {
#define X(sig, name, code, t1, t2, t0) dispatch_## code ##_## sig ##_## name,
    OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) dispatch_## code ##_## sig ##_## name,
    OPS_UNARY
#undef X
#define X(sig, name, code, t1, t0) dispatch_## code ##_## sig ##_## name,
    OPS_COPY
#undef X
};

TacVM::TacVM(size_t dim, uint8_t S_count, uint8_t V_count) : S_(S_count, 1) {
  static_assert(sizeof(TacVM::dispatch_table_) == sizeof(TacVM::DispatchFnPtr) * OPS_TAC_COUNT, "");
  V_.resize(dim, V_count);
}

void TacVM::display(std::ostream &os) {
  for (auto i = 0; i < S_.size(); i++) {
    os << "s" << i << " =\n    " << S_(i) << "\n";
  }
  for (auto i = 0; i < V_.cols(); i++) {
    os << "v" << i << " =\n    " << V_.col(i).transpose() << '\n';
  }
}

void TacProgram::display(std::ostream &os) const {
  size_t i = 0;
  size_t len = tac_.size();
  os << "space: scalar " << (int) S_slots_ << ", vector " << (int) V_slots_ << '\n';

  while (i < len) {
    uint8_t op = tac_[i];
    switch (ops::number_of_input(op)) {
      case 1: {
        uint8_t _1 = tac_[i + 1], _0 = tac_[i + 2];
        i += 3;
        auto t0_str = ops::prefix_of(ops::output_type(op));
        auto t1_str = ops::prefix_of(ops::input_type_1(op));
        os << t0_str << (int) _0 << " = " << ops::name(op)
           << "(" << t1_str << (int) _1 << ")\n";
        break;
      }
      case 2: {
        uint8_t _1 = tac_[i + 1], _2 = tac_[i + 2], _0 = tac_[i + 3];
        i += 4;
        auto t0_str = ops::prefix_of(ops::output_type(op));
        auto t1_str = ops::prefix_of(ops::input_type_1(op));
        auto t2_str = ops::prefix_of(ops::input_type_2(op));
        os << t0_str << (int) _0 << " = " << ops::name(op)
           << "(" << t1_str << (int) _1
           << ", " << t2_str << (int) _2 << ")\n";
        break;
      }
      default: {
        bug();
      }
    }
  }
}

void TacProgram::execute(TacVM &vm) const {
  size_t pc = 0;
  size_t tac_size = tac_.size();
  while (pc < tac_size) {
    pc += TacVM::dispatch_table_[tac_[pc]](vm, tac_, pc);
  }
}

}
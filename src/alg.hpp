#ifndef AUTODA_ALG_HPP
#define AUTODA_ALG_HPP

#include "prelude.hpp"
#include "dag.hpp"

namespace autoda {

// v[0]: x
// v[1]: x_adv

struct SInitMethod {
  fstd::dynarray<float> values_;

  explicit SInitMethod(uint8_t S_slots) : values_(S_slots) {}

  inline void run(TacVM &vm) const {
    memcpy(vm.S_.data(), values_.data(), sizeof(float) * values_.size());
  }
};

struct InitMethod {
  SInitMethod S_init_;
  fstd::dynarray<std::optional<float>> V_init_;

  InitMethod(uint8_t S_slots, uint8_t V_slots) : S_init_(S_slots), V_init_(V_slots) {}

  inline void run_S(TacVM &vm) const {
    S_init_.run(vm);
  }

  inline void run_V(TacVM &vm) const {
    for (size_t i = 0; i < V_init_.size(); i++) {
      if (V_init_[i].has_value()) {
        float value = V_init_[i].value();
        vm.V_.col(i).fill(value);
      }
    }
  }
};

struct AttackMethod {
  Dag dag_;
  uint16_t V_output_idx_;

  AttackMethod(uint8_t S_slots, uint8_t V_slots)
      : dag_(S_slots, V_slots), V_output_idx_(UINT16_MAX) {
  }

  void display(std::ostream &os) {
    os << "output: v" << V_output_idx_ << "\n";
    dag_.display(os);
  }
};

struct CompiledAttackMethod {
  TacProgram tac_;
  uint8_t V_output_idx_;

  explicit CompiledAttackMethod(AttackMethod &attack_method) {
    fstd::dynarray<uint8_t> S_outputs;
    fstd::dynarray<uint8_t> V_outputs(1);
    attack_method.dag_.compile<false>(tac_, {}, {attack_method.V_output_idx_},
                                      S_outputs, V_outputs);
    V_output_idx_ = V_outputs[0];
  }
};

struct LearnMethod {
  Dag dag_;
  std::vector<uint16_t> S_output_idxes_{};
  std::vector<uint16_t> V_output_idxes_{};

  LearnMethod(uint8_t S_slots, uint8_t V_slots)
      : dag_(S_slots, V_slots) {
  }
};

struct CompiledLearnMethod {
  TacProgram tac_;
  fstd::dynarray<uint8_t> S_output_idxes_;
  fstd::dynarray<uint8_t> V_output_idxes_;

  explicit CompiledLearnMethod(LearnMethod &learn_method)
      : S_output_idxes_(learn_method.S_output_idxes_.size()),
        V_output_idxes_(learn_method.V_output_idxes_.size()) {
    learn_method.dag_.compile<true>(
        tac_, learn_method.S_output_idxes_, learn_method.V_output_idxes_,
        S_output_idxes_, V_output_idxes_);
  }

  inline void run_S(TacVM &vm) const {
    for (size_t i = 0; i < S_output_idxes_.size(); i++) {
      uint8_t output = S_output_idxes_[i];
      if (i != output) { vm.S_(i) = vm.S_(output); }
    }
  }

  inline void run_V(TacVM &vm) const {
    for (size_t i = 0; i < V_output_idxes_.size(); i++) {
      uint8_t output = V_output_idxes_[i];
      if (i != output) { vm.V_.col(i) = vm.V_.col(output); }
    }
  }
};

struct Alg {
  InitMethod init_;
  AttackMethod attack_;
  LearnMethod learn_;

  Alg(uint8_t S_slots, uint8_t V_slots);

  // storage slots for scalars
  inline uint8_t S_slots() const noexcept { return init_.S_init_.values_.size(); }
  // storage slots for vectors
  inline uint8_t V_slots() const noexcept { return init_.V_init_.size(); }
};

struct CompiledAlg {
  CompiledAttackMethod attack_;
  CompiledLearnMethod learn_;

  explicit CompiledAlg(Alg &alg);
  CompiledAlg(const CompiledAlg &that) = delete;

  std::unique_ptr<TacVM> alloc_vm(size_t dim);
};

}

#endif //AUTODA_ALG_HPP

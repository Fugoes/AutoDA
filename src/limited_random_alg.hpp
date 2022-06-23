#ifndef AUTODA_LIMITED_RANDOM_ALG_HPP
#define AUTODA_LIMITED_RANDOM_ALG_HPP

#include "prelude.hpp"
#include "alg.hpp"
#include "random.hpp"
#include "executor.hpp"
#include "validator.hpp"

namespace autoda {

// Only scalar hyper-parameters are allowed in LimitedRandomAlg.
// V[0] = x; V[1] = x_adv; V[2] = noise; (V_slots == 3)

struct LimitedRandomAlgCfg {
  size_t dag_max_len_;
  std::vector<uint8_t> dag_ops_list_;
  uint8_t S_slots_;
  std::vector<float> allowed_S_values_;
};

struct LimitedRandomAlg {
  SInitMethod S_init_;
  AttackMethod attack_;
  std::optional<CompiledAttackMethod> compiled_attack_{};

  LimitedRandomAlg(const LimitedRandomAlgCfg &cfg)
      : S_init_(cfg.S_slots_), attack_(cfg.S_slots_, 3) {}

  inline void reset(const LimitedRandomAlgCfg &cfg) {
    attack_.V_output_idx_ = UINT16_MAX;
    auto &nodes = attack_.dag_.nodes_;
    attack_.dag_.nodes_.erase(nodes.begin() + cfg.S_slots_ + 3, nodes.end());
    compiled_attack_.reset();
  }

  bool generate_attack_naive(LimitedRandomAlgCfg &cfg);

  bool generate_attack_simple(LimitedRandomAlgCfg &cfg);

  bool generate_attack_compat(LimitedRandomAlgCfg &cfg);

  bool generate_attack_compact_wo_predefined_operations(LimitedRandomAlgCfg &cfg);

  // check if the attack use x, x_adv, noise, and all S[*].
  bool inputs_check(const LimitedRandomAlgCfg &cfg) {
    using T = uint8_t;
    BitOrValidator<T> validator(attack_.dag_);
    constexpr T bit_x = 1u << 0u;
    constexpr T bit_x_adv = 1u << 1u;
    constexpr T bit_noise = 1u << 2u;
#define bit_s_(_i) (1u << (3u + _i))
    validator.storage_[cfg.S_slots_ + 0] = bit_x;
    validator.storage_[cfg.S_slots_ + 1] = bit_x_adv;
    validator.storage_[cfg.S_slots_ + 2] = bit_noise;
    for (size_t i = 0; i < cfg.S_slots_; i++) { validator.storage_[i] = bit_s_(i); }
    size_t lo = cfg.S_slots_ + 3;
    if (!validator.run(attack_.dag_, lo)) return false;
    const T output = validator.storage_[attack_.V_output_idx_];
    bool has_x = output & bit_x;
    bool has_x_adv = output & bit_x_adv;
    bool has_noise = output & bit_noise;
    if (has_x && has_x_adv && has_noise) {
      for (size_t i = 0; i < cfg.S_slots_; i++) { if (!(output & bit_s_(i))) return false; }
      return true;
    } else {
      return false;
    }
  }

  void generate_init(const LimitedRandomAlgCfg &cfg) {
    for (auto &value : S_init_.values_) { value = random_select_from(cfg.allowed_S_values_); }
  }

  inline void compile() { compiled_attack_.emplace(attack_); }

  bool dist_check(const LimitedRandomAlgCfg &cfg, DistValidator &validator);
};

}

#endif //AUTODA_LIMITED_RANDOM_ALG_HPP

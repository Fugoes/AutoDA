#include "alg.hpp"

namespace autoda {

Alg::Alg(uint8_t S_slots, uint8_t V_slots) : init_(S_slots, V_slots),
                                             attack_(S_slots, V_slots),
                                             learn_(S_slots, V_slots) {
}

CompiledAlg::CompiledAlg(Alg &alg) : attack_(alg.attack_),
                                     learn_(alg.learn_) {
}

std::unique_ptr<TacVM> CompiledAlg::alloc_vm(size_t dim) {
  uint8_t S_slots = std::max(attack_.tac_.S_slots_, learn_.tac_.S_slots_);
  uint8_t V_slots = std::max(attack_.tac_.V_slots_, learn_.tac_.V_slots_);
  return std::make_unique<TacVM>(dim, S_slots, V_slots);
}

}

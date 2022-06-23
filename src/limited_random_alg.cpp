#include "limited_random_alg.hpp"

namespace {

using namespace autoda;

struct OpsListFilter {
  std::vector<uint8_t> &ops_list_;

  explicit OpsListFilter(LimitedRandomAlgCfg &cfg) : ops_list_(cfg.dag_ops_list_) {}
  inline uint8_t filter(std::function<bool(uint8_t)> cond) noexcept {
    const size_t ops_list_size = ops_list_.size();
    for (size_t i = 0; i < ops_list_size; i++) {
      std::uniform_int_distribution<size_t> dist(i, ops_list_size - 1);
      size_t target = dist(gen);
      std::swap(ops_list_[i], ops_list_[target]);
      auto op = ops_list_[i];
      if (cond(op)) return op;
    }
    return UINT8_MAX;
  }
  inline uint8_t get() noexcept {
    return random_select_from(ops_list_);
  }
};

// struct OpsListFilter {
//   std::vector<uint8_t> &ops_list_;
//   size_t pointer_{0};
//
//   explicit OpsListFilter(LimitedRandomAlgCfg &cfg) : ops_list_(cfg.dag_ops_list_) {
//     std::shuffle(ops_list_.begin(), ops_list_.end(), gen);
//   }
//   inline uint8_t filter(std::function<bool(uint8_t)> cond) noexcept {
//     size_t i;
//     for (i = pointer_; i < ops_list_.size(); i++) {
//       uint8_t op = ops_list_[i];
//       if (cond(op)) {
//         std::swap(ops_list_[pointer_], ops_list_[i]);
//         pointer_++;
//         return op;
//       }
//     }
//     return UINT8_MAX;
//   }
//   inline uint8_t get() noexcept {
//     if (pointer_ < ops_list_.size()) {
//       return ops_list_[pointer_++];
//     } else {
//       return UINT8_MAX;
//     }
//   }
// };

struct UnusedNodeMgr {
  std::vector<DagNode> &nodes_;
  uint16_t S_count{0}, V_count{0};

  inline UnusedNodeMgr(std::vector<DagNode> &nodes) : nodes_(nodes) {}
  inline uint16_t random_select(uint16_t idx, ops::DataType dt) {
    uint16_t count;
    if (dt == ops::DataType::Scalar) {
      if (S_count == 0) return UINT16_MAX;
      count = S_count;
    } else { // dt == ops::DataType::Vector
      if (V_count == 0) return UINT16_MAX;
      count = V_count;
    }
    std::uniform_int_distribution<uint16_t> target_dist(1, count);
    uint16_t target = target_dist(gen);
    uint16_t result = idx;
    for (uint16_t i = 0; i < target; i++) {
      while (true) {
        result--;
        auto &node = nodes_[result];
        if (node._flag_1 && ops::output_type(node.op_) == dt) break;
      }
    }
    auto &node = nodes_[result];
    node._flag_1 = false;
    if (dt == autoda::ops::DataType::Scalar) {
      S_count--;
    } else { // dt == autoda::ops::DataType::Vector
      V_count--;
    }
    return result;
  }
  inline void declare(uint16_t idx) {
    auto &node = nodes_[idx];
    node._flag_1 = true;
    if (ops::output_type(node.op_) == autoda::ops::DataType::Scalar) {
      S_count++;
    } else { // ops::output_type(node.op_) == autoda::ops::DataType::Vector
      V_count++;
    }
  }
};

}

namespace autoda {

bool LimitedRandomAlg::generate_attack_naive(LimitedRandomAlgCfg &cfg) {
  auto &dag = attack_.dag_;

  std::vector<uint16_t> S_idxes, V_idxes;
  /* setup hyper-parameters & parameters */ {
    for (uint16_t i = 0; i < cfg.S_slots_; i++) { S_idxes.push_back(i); }     // hyper-parameters
    for (uint16_t i = 0; i < 3; i++) { V_idxes.push_back(i + cfg.S_slots_); } // x, x_adv, noise
  }
  OpsListFilter ops_list(cfg);

#define fill_1(t) \
  if ((t) == ops::DataType::Scalar) { node._1 = random_select_from(S_idxes); } \
  else { node._1 = random_select_from(V_idxes); }
#define fill_2(t) \
  if ((t) == ops::DataType::Scalar) { node._2 = random_select_from(S_idxes); } \
  else { node._2 = random_select_from(V_idxes); }
#define declare(_op, _idx) \
  if (ops::output_type(_op) == ops::DataType::Scalar) { \
    S_idxes.push_back(_idx); \
  } else { \
    V_idxes.push_back(_idx); \
  }

  size_t dag_len = cfg.dag_max_len_;
  uint16_t defined_dag_len = dag.nodes_.size();
  for (auto i = defined_dag_len; i < dag_len; i++) { dag.emit(UINT8_MAX); }

  /* generate intermediate nodes */ {
    for (uint16_t idx = defined_dag_len; idx < dag_len - 1; idx++) {
      auto &node = dag.nodes_[idx];
      node.op_ = ops_list.get();
      if (node.op_ == UINT8_MAX) return false;
      if (ops::number_of_input(node.op_) == 1) {
        fill_1(ops::input_type_1(node.op_));
      } else { // ops::number_of_input(node.op_) == 2
        fill_1(ops::input_type_1(node.op_));
        fill_2(ops::input_type_2(node.op_));
      }
      declare(node.op_, idx);
    }
  }
  /* generate output node */ {
    size_t idx = dag_len - 1;
    auto &node = dag.nodes_[idx];
    node.op_ = ops_list.filter([](uint8_t op) {
      return ops::output_type(op) == ops::DataType::Vector;
    });
    if (node.op_ == UINT8_MAX) return false;
    if (ops::number_of_input(node.op_) == 1) {
      fill_1(ops::input_type_1(node.op_));
    } else { // ops::number_of_input(node.op_) == 2
      fill_1(ops::input_type_1(node.op_));
      fill_2(ops::input_type_2(node.op_));
    }
    declare(node.op_, idx);
    attack_.V_output_idx_ = idx;
  }

  return true;
#undef fill_1
#undef fill_2
#undef declare
}

bool LimitedRandomAlg::generate_attack_simple(LimitedRandomAlgCfg &cfg) {
  auto &dag = attack_.dag_;

  std::vector<uint16_t> S_idxes, V_idxes;
  /* setup hyper-parameters & parameters */ {
    for (uint16_t i = 0; i < cfg.S_slots_; i++) { S_idxes.push_back(i); }     // hyper-parameters
    for (uint16_t i = 0; i < 3; i++) { V_idxes.push_back(i + cfg.S_slots_); } // x, x_adv, noise
  }
  /* setup 3 predefined nodes */ {
    uint16_t x = V_idxes[0];
    uint16_t x_adv = V_idxes[1];
    auto src_direction = dag.emit(ops::VV_V_SUB, x, x_adv);
    auto src_direction_norm = dag.emit(ops::V_S_NORM, src_direction);
    auto src_direction_unit = dag.emit(ops::VS_V_DIV, src_direction, src_direction_norm);
    S_idxes.push_back(src_direction_norm);
    V_idxes.push_back(src_direction);
    V_idxes.push_back(src_direction_unit);
  }
  OpsListFilter ops_list(cfg);

#define fill_1(t) \
  if ((t) == ops::DataType::Scalar) { node._1 = random_select_from(S_idxes); } \
  else { node._1 = random_select_from(V_idxes); }
#define fill_2(t) \
  if ((t) == ops::DataType::Scalar) { node._2 = random_select_from(S_idxes); } \
  else { node._2 = random_select_from(V_idxes); }
#define declare(_op, _idx) \
  if (ops::output_type(_op) == ops::DataType::Scalar) { \
    S_idxes.push_back(_idx); \
  } else { \
    V_idxes.push_back(_idx); \
  }

  size_t dag_len = cfg.dag_max_len_;
  uint16_t defined_dag_len = dag.nodes_.size();
  for (auto i = defined_dag_len; i < dag_len; i++) { dag.emit(UINT8_MAX); }

  /* generate intermediate nodes */ {
    for (uint16_t idx = defined_dag_len; idx < dag_len - 1; idx++) {
      auto &node = dag.nodes_[idx];
      node.op_ = ops_list.get();
      if (node.op_ == UINT8_MAX) return false;
      if (ops::number_of_input(node.op_) == 1) {
        fill_1(ops::input_type_1(node.op_));
      } else { // ops::number_of_input(node.op_) == 2
        fill_1(ops::input_type_1(node.op_));
        fill_2(ops::input_type_2(node.op_));
      }
      declare(node.op_, idx);
    }
  }
  /* generate output node */ {
    size_t idx = dag_len - 1;
    auto &node = dag.nodes_[idx];
    node.op_ = ops_list.filter([](uint8_t op) {
      return ops::output_type(op) == ops::DataType::Vector;
    });
    if (node.op_ == UINT8_MAX) return false;
    if (ops::number_of_input(node.op_) == 1) {
      fill_1(ops::input_type_1(node.op_));
    } else { // ops::number_of_input(node.op_) == 2
      fill_1(ops::input_type_1(node.op_));
      fill_2(ops::input_type_2(node.op_));
    }
    declare(node.op_, idx);
    attack_.V_output_idx_ = idx;
  }

  return true;
#undef fill_1
#undef fill_2
#undef declare
}

bool LimitedRandomAlg::generate_attack_compact_wo_predefined_operations(LimitedRandomAlgCfg &cfg) {
  auto &dag = attack_.dag_;

  std::vector<uint16_t> S_idxes, V_idxes;
  /* setup hyper-parameters & parameters */ {
    for (uint16_t i = 0; i < cfg.S_slots_; i++) { S_idxes.push_back(i); }     // hyper-parameters
    for (uint16_t i = 0; i < 3; i++) { V_idxes.push_back(i + cfg.S_slots_); } // x, x_adv, noise
  }

  OpsListFilter ops_list(cfg);

  UnusedNodeMgr mgr(dag.nodes_);
  dag.nodes_[cfg.S_slots_ + 2]._flag_1 = true; // noise
  for (uint16_t i = 0; i < cfg.S_slots_; i++) { dag.nodes_[i]._flag_1 = true; } // S
  mgr.V_count = 1;
  mgr.S_count = cfg.S_slots_;

#define fill_1(t) \
  if ((t) == ops::DataType::Scalar) { node._1 = random_select_from(S_idxes); } \
  else { node._1 = random_select_from(V_idxes); }
#define fill_2(t) \
  if ((t) == ops::DataType::Scalar) { node._2 = random_select_from(S_idxes); } \
  else { node._2 = random_select_from(V_idxes); }
#define declare(_op, _idx) \
  if (ops::output_type(_op) == ops::DataType::Scalar) { \
    S_idxes.push_back(_idx); \
  } else { \
    V_idxes.push_back(_idx); \
  } \
  mgr.declare(_idx)

  size_t dag_len = cfg.dag_max_len_;
  uint16_t defined_dag_len = dag.nodes_.size();
  for (auto i = defined_dag_len; i < dag_len; i++) { dag.emit(UINT8_MAX); }

  /* generate intermediate nodes */ {
    for (uint16_t idx = defined_dag_len; idx < dag_len - 1; idx++) {
      auto &node = dag.nodes_[idx];
      node.op_ = ops_list.get();
      if (node.op_ == UINT8_MAX) return false;
      if (ops::number_of_input(node.op_) == 1) {
        std::uniform_int_distribution<unsigned> flag_dist(0, 1);
        // try to link _1 to unused node
        if (flag_dist(gen)) { node._1 = mgr.random_select(idx, ops::input_type_1(node.op_)); }
        // link _1 to random node
        if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
      } else { // ops::number_of_input(node.op_) == 2
        std::uniform_int_distribution<unsigned> n_dist(0, 2);
        auto n = n_dist(gen);
        if (n == 2) { // try to link _1 and _2 to unused nodes
          node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
          node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
        } else if (n == 1) { // try to link _1 or _2 to unused nodes
          std::uniform_int_distribution<unsigned> flag_dist(0, 1);
          if (flag_dist(gen)) {
            node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
          } else {
            node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
          }
        }
        // link _1 to random node
        if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
        // link _2 to random node
        if (node._2 == UINT16_MAX) { fill_2(ops::input_type_2(node.op_)); }
      }
      declare(node.op_, idx);
    }
  }
  /* generate output node */ {
    size_t idx = dag_len - 1;
    auto &node = dag.nodes_[idx];
    node.op_ = ops_list.filter([](uint8_t op) {
      return ops::output_type(op) == ops::DataType::Vector;
    });
    if (node.op_ == UINT8_MAX) return false;
    if (ops::number_of_input(node.op_) == 1) {
      std::uniform_int_distribution<unsigned> flag_dist(0, 1);
      // try to link _1 to unused node
      if (flag_dist(gen)) { node._1 = mgr.random_select(idx, ops::input_type_1(node.op_)); }
      // link _1 to random node
      if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
    } else { // ops::number_of_input(node.op_) == 2
      std::uniform_int_distribution<unsigned> n_dist(0, 2);
      auto n = n_dist(gen);
      if (n == 2) { // try to link _1 and _2 to unused nodes
        node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
        node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
      } else if (n == 1) { // try to link _1 or _2 to unused node
        std::uniform_int_distribution<unsigned> flag_dist(0, 1);
        if (flag_dist(gen)) {
          node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
        } else {
          node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
        }
      }
      // link _1 to random node
      if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
      // link _2 to random node
      if (node._2 == UINT16_MAX) { fill_2(ops::input_type_2(node.op_)); }
    }
    declare(node.op_, idx);
    attack_.V_output_idx_ = idx;
  }

  return true;
#undef fill_1
#undef fill_2
#undef declare
}

bool LimitedRandomAlg::generate_attack_compat(LimitedRandomAlgCfg &cfg) {
  auto &dag = attack_.dag_;

  std::vector<uint16_t> S_idxes, V_idxes;
  /* setup hyper-parameters & parameters */ {
    for (uint16_t i = 0; i < cfg.S_slots_; i++) { S_idxes.push_back(i); }     // hyper-parameters
    for (uint16_t i = 0; i < 3; i++) { V_idxes.push_back(i + cfg.S_slots_); } // x, x_adv, noise
  }
  /* setup 3 predefined nodes */ {
    uint16_t x = V_idxes[0];
    uint16_t x_adv = V_idxes[1];
    auto src_direction = dag.emit(ops::VV_V_SUB, x, x_adv);
    auto src_direction_norm = dag.emit(ops::V_S_NORM, src_direction);
    auto src_direction_unit = dag.emit(ops::VS_V_DIV, src_direction, src_direction_norm);
    S_idxes.push_back(src_direction_norm);
    V_idxes.push_back(src_direction);
    V_idxes.push_back(src_direction_unit);
  }

  OpsListFilter ops_list(cfg);

  UnusedNodeMgr mgr(dag.nodes_);
  dag.nodes_[cfg.S_slots_ + 2]._flag_1 = true; // noise
  for (uint16_t i = 0; i < cfg.S_slots_; i++) { dag.nodes_[i]._flag_1 = true; } // S
  mgr.V_count = 1;
  mgr.S_count = cfg.S_slots_;

#define fill_1(t) \
  if ((t) == ops::DataType::Scalar) { node._1 = random_select_from(S_idxes); } \
  else { node._1 = random_select_from(V_idxes); }
#define fill_2(t) \
  if ((t) == ops::DataType::Scalar) { node._2 = random_select_from(S_idxes); } \
  else { node._2 = random_select_from(V_idxes); }
#define declare(_op, _idx) \
  if (ops::output_type(_op) == ops::DataType::Scalar) { \
    S_idxes.push_back(_idx); \
  } else { \
    V_idxes.push_back(_idx); \
  } \
  mgr.declare(_idx)

  size_t dag_len = cfg.dag_max_len_;
  uint16_t defined_dag_len = dag.nodes_.size();
  for (auto i = defined_dag_len; i < dag_len; i++) { dag.emit(UINT8_MAX); }

  /* generate intermediate nodes */ {
    for (uint16_t idx = defined_dag_len; idx < dag_len - 1; idx++) {
      auto &node = dag.nodes_[idx];
      node.op_ = ops_list.get();
      if (node.op_ == UINT8_MAX) return false;
      if (ops::number_of_input(node.op_) == 1) {
        std::uniform_int_distribution<unsigned> flag_dist(0, 1);
        // try to link _1 to unused node
        if (flag_dist(gen)) { node._1 = mgr.random_select(idx, ops::input_type_1(node.op_)); }
        // link _1 to random node
        if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
      } else { // ops::number_of_input(node.op_) == 2
        std::uniform_int_distribution<unsigned> n_dist(0, 2);
        auto n = n_dist(gen);
        if (n == 2) { // try to link _1 and _2 to unused nodes
          node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
          node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
        } else if (n == 1) { // try to link _1 or _2 to unused nodes
          std::uniform_int_distribution<unsigned> flag_dist(0, 1);
          if (flag_dist(gen)) {
            node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
          } else {
            node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
          }
        }
        // link _1 to random node
        if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
        // link _2 to random node
        if (node._2 == UINT16_MAX) { fill_2(ops::input_type_2(node.op_)); }
      }
      declare(node.op_, idx);
    }
  }
  /* generate output node */ {
    size_t idx = dag_len - 1;
    auto &node = dag.nodes_[idx];
    node.op_ = ops_list.filter([](uint8_t op) {
      return ops::output_type(op) == ops::DataType::Vector;
    });
    if (node.op_ == UINT8_MAX) return false;
    if (ops::number_of_input(node.op_) == 1) {
      std::uniform_int_distribution<unsigned> flag_dist(0, 1);
      // try to link _1 to unused node
      if (flag_dist(gen)) { node._1 = mgr.random_select(idx, ops::input_type_1(node.op_)); }
      // link _1 to random node
      if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
    } else { // ops::number_of_input(node.op_) == 2
      std::uniform_int_distribution<unsigned> n_dist(0, 2);
      auto n = n_dist(gen);
      if (n == 2) { // try to link _1 and _2 to unused nodes
        node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
        node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
      } else if (n == 1) { // try to link _1 or _2 to unused node
        std::uniform_int_distribution<unsigned> flag_dist(0, 1);
        if (flag_dist(gen)) {
          node._1 = mgr.random_select(idx, ops::input_type_1(node.op_));
        } else {
          node._2 = mgr.random_select(idx, ops::input_type_2(node.op_));
        }
      }
      // link _1 to random node
      if (node._1 == UINT16_MAX) { fill_1(ops::input_type_1(node.op_)); }
      // link _2 to random node
      if (node._2 == UINT16_MAX) { fill_2(ops::input_type_2(node.op_)); }
    }
    declare(node.op_, idx);
    attack_.V_output_idx_ = idx;
  }

  return true;
#undef fill_1
#undef fill_2
#undef declare
}

bool LimitedRandomAlg::dist_check(const LimitedRandomAlgCfg &cfg, DistValidator &validator) {
  const size_t count = validator.vms_.size();
  size_t passed = validator.run(S_init_, *compiled_attack_);
  return passed == count;
}

}

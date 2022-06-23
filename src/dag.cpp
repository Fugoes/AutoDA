#include "dag.hpp"

namespace {

inline void update_lifetime(autoda::__detail::DagCompiler &compiler,
                            uint16_t idx, uint16_t lifetime) {
  if (lifetime > compiler.lifetimes_[idx]) {
    compiler.lifetimes_[idx] = lifetime;
  }
}

inline void gc_and_calc_lifetimes(autoda::__detail::DagCompiler &compiler,
                                  autoda::Dag &dag,
                                  uint16_t root) {
  compiler.lifetimes_[root] = UINT16_MAX;

  std::stack<uint16_t> stk{};
  stk.push(root);

  while (!stk.empty()) {
    uint16_t idx = stk.top();
    stk.pop();
    auto &node = dag.nodes_[idx];
    update_lifetime(compiler, idx, idx);
    if (!node.gc_flag_ && !node.is_input()) {
      node.gc_flag_ = true;
      switch (autoda::ops::number_of_input(node.op_)) {
        case 1: {
          stk.push(node._1);
          update_lifetime(compiler, node._1, idx);
          break;
        }
        case 2: {
          stk.push(node._1);
          update_lifetime(compiler, node._1, idx);
          stk.push(node._2);
          update_lifetime(compiler, node._2, idx);
          break;
        }
        default: {
          autoda::bug();
        }
      }
    }
  }
}

}

namespace autoda {

Dag::Dag(uint16_t S_count, uint16_t V_count) {
  for (uint16_t i = 0; i < S_count; i++) {
    append(ops::S_INPUT);
  }
  for (uint16_t i = 0; i < V_count; i++) {
    append(ops::V_INPUT);
  }
}

void Dag::append(uint8_t op, uint16_t _1, uint16_t _2) {
  check(op != ops::S_S_COPY && op != ops::V_V_COPY);
  nodes_.emplace_back(op, _1, _2);
  size_t idx = nodes_.size() - 1;
  check(idx <= UINT16_MAX - 1);
}

void Dag::display(std::ostream &os) const {
  for (uint16_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i];
    auto t0_str = ops::prefix_of(ops::output_type(node.op_));
    os << t0_str << i << " = " << ops::name(nodes_[i].op_) << "(";
    switch (ops::number_of_input(node.op_)) {
      case 0: {
        break;
      }
      case 1: {
        const char *t1_str = ops::prefix_of(ops::input_type_1(node.op_));
        os << t1_str << node._1;
        break;
      }
      case 2: {
        const char *t1_str = ops::prefix_of(ops::input_type_1(node.op_));
        const char *t2_str = ops::prefix_of(ops::input_type_2(node.op_));
        os << t1_str << node._1 << ", " << t2_str << node._2;
        break;
      }
    }
    os << ")\n";
  }
}

void Dag::display_graphviz(std::ostream &os) const {
  os << "digraph {\n" << "  graph [ordering=\"out\"];\n" << "  clusterrank=local;\n";
  std::vector<std::string> inputs;
  for (uint16_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i];
    auto t0_str = ops::prefix_of(ops::output_type(node.op_)) + std::to_string(i) +
        "_" + ops::fullname(node.op_);
    switch (ops::number_of_input(node.op_)) {
      case 0: {
        inputs.push_back(t0_str);
        break;
      }
      case 1: {
        auto t1_str = ops::prefix_of(ops::output_type(nodes_[node._1].op_)) +
            std::to_string(node._1) + "_" + ops::fullname(nodes_[node._1].op_);
        os << "  " << t1_str << " -> " << t0_str << "\n";
        break;
      }
      case 2: {
        auto t1_str = ops::prefix_of(ops::output_type(nodes_[node._1].op_)) +
            std::to_string(node._1) + "_" + ops::fullname(nodes_[node._1].op_);
        auto t2_str = ops::prefix_of(ops::output_type(nodes_[node._2].op_)) +
            std::to_string(node._2) + "_" + ops::fullname(nodes_[node._2].op_);
        os << "  " << t1_str << " -> " << t0_str << "\n";
        os << "  " << t2_str << " -> " << t0_str << "\n";
        break;
      }
    }
  }
  os << "  subgraph INPUTS {\n" << "    rank=same;\n";
  for (auto &input : inputs) { os << "    " << input << "\n"; }
  os << "  }\n" << "}\n";
}

void Dag::type_check() const {
  for (uint16_t idx = 0; idx < nodes_.size(); idx++) {
    auto &node = nodes_[idx];
    switch (ops::number_of_input(node.op_)) {
      case 0: {
        break;
      }
      case 1: {
        check(node._1 < idx);
        check(ops::output_type(nodes_[node._1].op_) == ops::input_type_1(node.op_));
        break;
      }
      case 2: {
        check(node._1 < idx);
        check(ops::output_type(nodes_[node._1].op_) == ops::input_type_1(node.op_));
        check(node._2 < idx);
        check(ops::output_type(nodes_[node._2].op_) == ops::input_type_2(node.op_));
        break;
      }
      default: {
        bug();
      }
    }
  }
}

DagWriter::DagWriter(Dag &dag) : dag_(dag) {}

#define X(sig, name, code, t1, t2, t0) \
DagNodeSymbol ## t0 DagWriter::name(DagNodeSymbol ## t1 _1, DagNodeSymbol ## t2 _2) { \
  dag_.append(ops::sig ##_## name, _1.idx_, _2.idx_); \
  return DagNodeSymbol ## t0(dag_.nodes_.size() - 1); \
}
OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) \
DagNodeSymbol ## t0 DagWriter::name(DagNodeSymbol ## t1 _1) { \
  dag_.append(ops::sig ##_## name, _1.idx_); \
  return DagNodeSymbol ## t0(dag_.nodes_.size() - 1); \
}
OPS_UNARY
#undef X

namespace __detail {

DagCompiler::DagCompiler(Dag &dag,
                         const std::vector<uint16_t> &S_wanted,
                         const std::vector<uint16_t> &V_wanted) :
    S_wanted_(S_wanted),
    V_wanted_(V_wanted),
    dag_idx_to_tac_idx_(dag.nodes_.size(), UINT8_MAX),
    lifetimes_(dag.nodes_.size(), 0) {
  for (uint16_t idx = 0; idx < dag.nodes_.size(); idx++) {
    auto &node = dag.nodes_[idx];
    if (!node.is_input()) break;
    lifetimes_[idx] = UINT16_MAX;
  }

  for (uint16_t root : S_wanted) {
    gc_and_calc_lifetimes(*this, dag, root);
  }
  for (uint16_t root : V_wanted) {
    gc_and_calc_lifetimes(*this, dag, root);
  }
}

void DagCompiler::codegen(TacProgram &program, Dag &dag) {
  uint16_t len = dag.nodes_.size();
  for (uint16_t idx = 0; idx < len; idx++) {
    auto &node = dag.nodes_[idx];
    if (node.gc_flag_ || node.is_input()) {
      node.gc_flag_ = false;
      uint8_t _0;
      if (ops::output_type(node.op_) == ops::DataType::Scalar) {
        _0 = S_mgr_.alloc(idx, lifetimes_[idx]);
      } else { // ops::output_type(node.op_) == ops::DataType::Vector
        _0 = V_mgr_.alloc(idx, lifetimes_[idx]);
      }
      dag_idx_to_tac_idx_[idx] = _0;
      switch (ops::number_of_input(node.op_)) {
        case 0: {
          break;
        }
        case 1: {
          uint8_t _1 = dag_idx_to_tac_idx_[node._1];
          check(_1 != UINT8_MAX);
          program.emit(node.op_, _1, _0);
          break;
        }
        case 2: {
          uint8_t _1 = dag_idx_to_tac_idx_[node._1];
          check(_1 != UINT8_MAX);
          uint8_t _2 = dag_idx_to_tac_idx_[node._2];
          check(_2 != UINT8_MAX);
          program.emit(node.op_, _1, _2, _0);
          break;
        }
        default: {
          bug();
        }
      }
    }
  }
}

void DagCompiler::copy_outputs(TacProgram &program, Dag &dag,
                               fstd::dynarray<uint8_t> &S_outputs,
                               fstd::dynarray<uint8_t> &V_outputs) {
  for (size_t i = 0; i < S_wanted_.size(); i++) {
    uint8_t _0;
    uint16_t idx = S_wanted_[i];
    if (dag.nodes_[idx].is_input() && i != idx) {
      _0 = S_mgr_.alloc(idx, UINT16_MAX);
      uint8_t _1 = dag_idx_to_tac_idx_[idx];
      program.emit(ops::S_S_COPY, _1, _0);
    } else {
      _0 = dag_idx_to_tac_idx_[idx];
    }
    S_outputs[i] = _0;
  }
  for (size_t i = 0; i < V_wanted_.size(); i++) {
    uint8_t _0;
    uint16_t idx = V_wanted_[i];
    if (dag.nodes_[idx].is_input() && i + S_wanted_.size() != idx) {
      _0 = V_mgr_.alloc(idx, UINT16_MAX);
      uint8_t _1 = dag_idx_to_tac_idx_[idx];
      program.emit(ops::V_V_COPY, _1, _0);
    } else {
      _0 = dag_idx_to_tac_idx_[idx];
    }
    V_outputs[i] = _0;
  }
}

void DagCompiler::no_copy_outputs(fstd::dynarray<uint8_t> &S_outputs,
                                  fstd::dynarray<uint8_t> &V_outputs) {
  for (size_t i = 0; i < S_wanted_.size(); i++) {
    S_outputs[i] = dag_idx_to_tac_idx_[S_wanted_[i]];
  }
  for (size_t i = 0; i < V_wanted_.size(); i++) {
    V_outputs[i] = dag_idx_to_tac_idx_[V_wanted_[i]];
  }
}

}

}

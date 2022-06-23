#ifndef AUTODA_DAG_HPP
#define AUTODA_DAG_HPP

#include "prelude.hpp"

#include "ops.hpp"
#include "tac.hpp"

namespace autoda {

struct Dag;

namespace __detail {

struct SlotsManager {
  std::vector<uint16_t> slots_lifetimes_{};

  // Alloc one slot given current time t with lifetime. A node's lifetime means when t >= lifetime,
  // the node's output is no longer needed.
  inline uint8_t alloc(uint16_t t, uint16_t lifetime) {
    for (uint8_t i = 0; i < slots_lifetimes_.size(); i++) {
      if (t >= slots_lifetimes_[i]) {
        slots_lifetimes_[i] = lifetime;
        return i;
      }
    }
    slots_lifetimes_.push_back(lifetime);
    autoda::check(slots_lifetimes_.size() < UINT8_MAX);
    return slots_lifetimes_.size() - 1;
  }
};

struct DagCompiler {
  const std::vector<uint16_t> &S_wanted_;
  const std::vector<uint16_t> &V_wanted_;
  SlotsManager S_mgr_{};
  SlotsManager V_mgr_{};
  fstd::dynarray<uint8_t> dag_idx_to_tac_idx_;
  fstd::dynarray<uint16_t> lifetimes_;

  DagCompiler(Dag &dag,
              const std::vector<uint16_t> &S_wanted, const std::vector<uint16_t> &V_wanted);

  void codegen(TacProgram &program, Dag &dag);

  void copy_outputs(TacProgram &program, Dag &dag,
                    fstd::dynarray<uint8_t> &S_outputs, fstd::dynarray<uint8_t> &V_outputs);

  void no_copy_outputs(fstd::dynarray<uint8_t> &S_outputs, fstd::dynarray<uint8_t> &V_outputs);
};

}

// It is more like some SSA form. Each instruction generate a new variable identified by its index.
// We prefer a fixed length representation here. To handle input nodes consistently with normal
// nodes, we add extra ops for input.
//
// OPS_*_COPY are not valid in DAG.
struct DagNode {
  // gc_flag_ is for dead code removal (a.k.a gc the DAG).
  uint8_t gc_flag_: 1;
  uint8_t _flag_1: 1;
  uint8_t op_;
  uint16_t _1;
  uint16_t _2;

  DagNode(uint8_t op, uint16_t _1, uint16_t _2)
      : gc_flag_(false), _flag_1(false), op_(op), _1(_1), _2(_2) {}

  inline bool is_input() noexcept {
    return op_ == ops::S_INPUT || op_ == ops::V_INPUT;
  }
};

static_assert(sizeof(DagNode) == 6, "sizeof(DagNode) == 6");

struct Dag {
  std::vector<DagNode> nodes_{};

  Dag(uint16_t S_count, uint16_t V_count);

  void append(uint8_t op, uint16_t _1 = UINT16_MAX, uint16_t _2 = UINT16_MAX);

  inline uint16_t emit(uint8_t op, uint16_t _1 = UINT16_MAX, uint16_t _2 = UINT16_MAX) {
    nodes_.emplace_back(op, _1, _2);
    return nodes_.size() - 1;
  }

  // Compile the DAG into TAC program. S_wanted and V_wanted are indexes of wanted nodes in DAG.
  // Their indexes in the TAC program are recorded in the {S,V}_output. If CopyInput, generate
  // extra *_*_COPY TAC if needed to avoid directly use inputs as output.
  template<bool CopyInput>
  void compile(TacProgram &program,
               const std::vector<uint16_t> &S_wanted, const std::vector<uint16_t> &V_wanted,
               fstd::dynarray<uint8_t> &S_outputs, fstd::dynarray<uint8_t> &V_outputs);

  void display(std::ostream &os) const;

  void display_graphviz(std::ostream &os) const;

  void type_check() const;
};

struct DagNodeSymbolScalar {
  uint16_t idx_;

  DagNodeSymbolScalar() : idx_(UINT16_MAX) {}

  explicit DagNodeSymbolScalar(uint16_t idx) : idx_(idx) {}
};

struct DagNodeSymbolVector {
  uint16_t idx_;

  DagNodeSymbolVector() : idx_(UINT16_MAX) {}

  explicit DagNodeSymbolVector(uint16_t idx) : idx_(idx) {}
};

struct DagWriter {
  Dag &dag_;

  DagWriter(Dag &dag);

#define X(sig, name, code, t1, t2, t0) \
  DagNodeSymbol ## t0 name(DagNodeSymbol ## t1 _1, DagNodeSymbol ## t2 _2);
  OPS_BINARY
#undef X
#define X(sig, name, code, t1, t0) \
  DagNodeSymbol ## t0 name(DagNodeSymbol ## t1 _1);
  OPS_UNARY
#undef X
};

template<bool CopyInput>
void Dag::compile(TacProgram &program,
                  const std::vector<uint16_t> &S_wanted, const std::vector<uint16_t> &V_wanted,
                  fstd::dynarray<uint8_t> &S_outputs, fstd::dynarray<uint8_t> &V_outputs) {
  __detail::DagCompiler compiler(*this, S_wanted, V_wanted);
  compiler.codegen(program, *this);
  if constexpr (CopyInput) {
    compiler.copy_outputs(program, *this, S_outputs, V_outputs);
  } else {
    compiler.no_copy_outputs(S_outputs, V_outputs);
  }
  program.S_slots_ = compiler.S_mgr_.slots_lifetimes_.size();
  program.V_slots_ = compiler.V_mgr_.slots_lifetimes_.size();
}

}

#endif //AUTODA_DAG_HPP

#ifndef AUTODA_BOUNDARY_ATTACK_HPP
#define AUTODA_BOUNDARY_ATTACK_HPP

#include "alg.hpp"

std::unique_ptr<autoda::Alg> boundary_attack() {
  std::unique_ptr<autoda::Alg> boundary{new autoda::Alg(3, 3)};
  // scalar storage
  autoda::DagNodeSymbolScalar src_step(0);
  autoda::DagNodeSymbolScalar sphe_step(1);
  autoda::DagNodeSymbolScalar factor(2);
  // vector storage
  autoda::DagNodeSymbolVector x(3);
  autoda::DagNodeSymbolVector x_adv(4);
  autoda::DagNodeSymbolVector n(5);

  /* init() */ {
    auto &S_init = boundary->init_.S_init_.values_;
    S_init[0] = 0.01; // src_step
    S_init[1] = 0.01; // sphe_step
    S_init[2] = sqrtf(1 + 0.01 * 0.01); // factor
    auto &V_init = boundary->init_.V_init_;
    V_init[0] = 0.0; // x
    V_init[1] = 0.0; // x_adv
    V_init[2] = 0.0; // n
  }

  /* attack() */ {
    autoda::DagWriter w(boundary->attack_.dag_);
    auto src_direction = w.SUB(x, x_adv);
    auto src_direction_norm = w.NORM(src_direction);
    auto src_direction_unit = w.DIV(src_direction, src_direction_norm);
    auto pert = n;
    pert = w.SUB(pert, w.MUL(src_direction_unit, w.DOT(pert, src_direction_unit)));
    pert = w.MUL(pert, w.DIV(w.MUL(sphe_step, src_direction_norm), w.NORM(pert)));
    auto new_src_direction = w.DIV(w.SUB(src_direction, pert), factor);
    auto new_src_direction_norm = w.NORM(new_src_direction);
    auto sphe_candidate = w.SUB(x, new_src_direction);
    auto length = w.MUL(src_step, src_direction_norm);
    auto deviation = w.SUB(new_src_direction_norm, src_direction_norm);
    length = w.DIV(w.ADD(length, deviation), new_src_direction_norm);
    auto candidate = w.ADD(sphe_candidate, w.MUL(new_src_direction, length));
    boundary->attack_.V_output_idx_ = candidate.idx_;
  }

  return boundary;
}

#endif //AUTODA_BOUNDARY_ATTACK_HPP

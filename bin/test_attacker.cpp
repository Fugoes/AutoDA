#include "alg.hpp"

// A simple random attacker
//
//  def init():
//      s0 = 0.00
//      s1 = 0.01
//  def attack():
//      v1 = norm(s0, s1)
//  def learn():
//      pass

const unsigned DIM = 10;

int main(int argc, char *argv[]) {
  if (argc != 2 || chdir(argv[1]) < 0) {
    perror("chdir()");
    std::exit(-1);
  }

  autoda::Alg attacker(2, 1);
  auto &S_init = attacker.init_.S_init_.values_;
  S_init[0] = 0.5;
  S_init[1] = 1.5;
  attacker.init_.V_init_[0] = 0.5;

  {
    autoda::DagWriter writer(attacker.attack_.dag_);
    auto x = writer.ADD(autoda::DagNodeSymbolVector(2), autoda::DagNodeSymbolScalar(1));
    auto y = writer.MUL(x, autoda::DagNodeSymbolScalar(0));
    auto z = writer.MUL(y, autoda::DagNodeSymbolScalar(0));
    attacker.attack_.V_output_idx_ = z.idx_;
  }

  {
    autoda::DagWriter writer(attacker.learn_.dag_);
    auto x = writer.ADD(autoda::DagNodeSymbolScalar(0), autoda::DagNodeSymbolScalar(1));
    auto y = writer.SUB(autoda::DagNodeSymbolVector(2), x);
    auto z = writer.RELU(y);
    attacker.learn_.S_output_idxes_ = {x.idx_, 1};
    attacker.learn_.V_output_idxes_ = {z.idx_};
  }

  autoda::CompiledAlg compiled_attacker(attacker);
  std::unique_ptr<autoda::TacVM> vm = compiled_attacker.alloc_vm(DIM);

  attacker.learn_.dag_.display(std::cout);
  std::cout << '\n';
  compiled_attacker.learn_.tac_.display(std::cout);
  std::cout << '\n';

  attacker.init_.run_S(*vm);
  attacker.init_.run_V(*vm);
  vm->display(std::cout);
  std::cout << '\n';

  Eigen::Matrix<float, Eigen::Dynamic, 1> noise(DIM, 1);
  compiled_attacker.attack_.tac_.execute(*vm);
  float *x_adv = vm->V_.col(compiled_attacker.attack_.V_output_idx_).data();
  vm->display(std::cout);
  std::cout << '\n';

  compiled_attacker.learn_.run_S(*vm);
  compiled_attacker.learn_.run_V(*vm);
  vm->display(std::cout);
  std::cout << '\n';
}
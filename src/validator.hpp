#ifndef AUTODA_VALIDATOR_HPP
#define AUTODA_VALIDATOR_HPP

#include "dag.hpp"
#include "tac.hpp"
#include "alg.hpp"
#include "random.hpp"

namespace autoda {

template<typename T>
struct SymbolicValidator {
  fstd::dynarray<T> storage_;

  using DispatchFn = bool (*)(decltype(storage_) &storage,
                              size_t idx, uint8_t op, uint16_t _1, uint16_t _2);

  explicit SymbolicValidator(size_t slots) : storage_(slots) {}

  template<DispatchFn DISPATCH>
  inline bool run_symbolic(const Dag &dag, size_t lo) {
    for (size_t i = lo; i < dag.nodes_.size(); i++) {
      const auto &node = dag.nodes_[i];
      if (!DISPATCH(storage_, i, node.op_, node._1, node._2)) return false;
    }
    return true;
  }
};

template<typename T>
struct BitOrValidator : public SymbolicValidator<T> {
  explicit BitOrValidator(const Dag &dag) : SymbolicValidator<T>(dag.nodes_.size()) {}

  bool run(const Dag &dag, size_t lo) {
    return SymbolicValidator<T>::template run_symbolic<dispatch>(dag, lo);
  }

 private:
  static bool dispatch(decltype(SymbolicValidator<T>::storage_) &storage,
                       size_t idx, uint8_t op, uint16_t _1, uint16_t _2) {
    switch (ops::number_of_input(op)) {
      case 1: {
        storage[idx] = storage[_1];
        return true;
      }
      case 2: {
        storage[idx] = storage[_1] | storage[_2];
        return true;
      }
      default: {
        return false;
      }
    }
  }
};

struct DistValidator {
  fstd::dynarray<float> dists_;
  std::vector<TacVM> vms_{}; // trade space for time

  DistValidator(unsigned count, size_t dim, uint8_t S_count, uint8_t V_count) : dists_(count) {
    vms_.reserve(count);
    for (unsigned i = 0; i < count; i++) { vms_.emplace_back(dim, S_count, V_count); }
  }

  // assume x is already refreshed
  template<size_t X_IDX = 0, size_t X_ADV_IDX = 1, size_t NOISE_IDX = 2>
  void refresh() {
    const size_t dim = vms_[0].V_.rows();
    const float sqrt_dim = sqrtf(dim);

    size_t count = vms_.size();
    float norm = 0.5, d = (15.0 - 0.5) / (count - 1);
    for (size_t i = 0; i < count; i++) {
      auto &vm = vms_[i];
      autoda_fill_randn(vm.V_.col(X_ADV_IDX));
      vm.V_.col(X_ADV_IDX) *= norm / sqrt_dim;
      vm.V_.col(X_ADV_IDX) += vm.V_.col(X_IDX);
      dists_[i] = autoda_norm(vm.V_.col(X_ADV_IDX) - vm.V_.col(X_IDX));
      autoda_fill_randn(vm.V_.col(NOISE_IDX));
      norm += d;
    }
  }

  template<size_t X_IDX = 0>
  size_t run(const SInitMethod &S_init, const CompiledAttackMethod &attack) {
    const size_t count = vms_.size();
    for (size_t passed = 0; passed < count; passed++) {
      auto &vm = vms_[passed];
      S_init.run(vm);
      attack.tac_.execute(vm);
      // do NOT clip the output here
      float dist = autoda_norm(vm.V_.col(attack.V_output_idx_) - vm.V_.col(X_IDX));
      if (!(dist < dists_[passed] && dist / dists_[passed] > 0.8)) return passed;
    }
    return count;
  }
};

struct SimulationValidator {
  size_t count_;
  TacVM vm_;

  SimulationValidator(unsigned count, size_t dim, uint8_t S_count, uint8_t V_count)
      : count_(count), vm_(dim, S_count, V_count) {
  }

  inline size_t run_simulation(std::function<void(TacVM & , size_t)> &&init_fn,
                               const TacProgram &program,
                               std::function<bool(const TacVM &, size_t)> &&validate_fn) {
    size_t passed;
    for (passed = 0; passed < count_; passed++) {
      init_fn(vm_, passed);
      program.execute(vm_);
      if (!validate_fn(vm_, passed)) { return passed; }
    }
    return count_;
  }
};

template<unsigned x_dim>
struct DistanceValidator : public SimulationValidator {
  Eigen::Array<float, x_dim, Eigen::Dynamic> xs_;
  Eigen::Array<float, x_dim, Eigen::Dynamic> starting_points_;
  fstd::dynarray<float> dists_;

  DistanceValidator(uint8_t S_count, uint8_t V_count, size_t n)
      : SimulationValidator(n, x_dim, S_count, V_count),
        xs_(x_dim, n), starting_points_(x_dim, n), dists_(n) {
  }

  void calculate_dists() {
    for (auto i = 0; i < xs_.cols(); i++) {
      dists_[i] = (xs_.col(i) - starting_points_.col(i)).matrix().norm();
    }
  }

  size_t validate(const InitMethod &init, const CompiledAttackMethod &attack) {
    return run_simulation(
        [this, &init](TacVM &vm, size_t i) {
          init.run_S(vm);
          vm.V_.col(0) = xs_.col(i);
          vm.V_.col(1) = starting_points_.col(i);
        },
        attack.tac_,
        [this, &attack](const TacVM &vm, size_t i) {
          // do NOT clip the output here
          float dist = (vm.V_.col(attack.V_output_idx_) - vm.V_.col(0)).matrix().norm();
          return dist < dists_[i] && dist / dists_[i] > 0.8;
        }
    );
  }
};

}

#endif //AUTODA_VALIDATOR_HPP

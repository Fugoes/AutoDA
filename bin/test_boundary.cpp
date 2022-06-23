#include "boundary_attack.hpp"
#include "models.hpp"
#include <map>

int main(int argc, char *argv[]) {
  if (argc != 2 || chdir(argv[1]) < 0) {
    perror("chdir()");
    std::exit(-1);
  }
  autoda::cifar2::initialize();

  std::unique_ptr<autoda::Alg> boundary = boundary_attack();
  std::unique_ptr<autoda::CompiledAlg> compiled_boundary(new autoda::CompiledAlg(*boundary));
  for (auto &node : boundary->attack_.dag_.nodes_) {
    std::cout << autoda::ops::fullname(node.op_) << std::endl;
  }
  std::cout << "\n";
  boundary->attack_.display(std::cout);
  std::cout << "\n";
  boundary->attack_.dag_.display_graphviz(std::cout);
  std::cout << "\n";
  compiled_boundary->attack_.tac_.display(std::cout);
  std::cout << "\n";

  boundary->learn_.dag_.display(std::cout);
  std::cout << "\n";
  boundary->learn_.dag_.display_graphviz(std::cout);
  std::cout << "\n";
  compiled_boundary->learn_.tac_.display(std::cout);
  std::cout << "\n";

  std::map<uint8_t, size_t> ops;
  for (auto &node : boundary->attack_.dag_.nodes_) {
    auto it = ops.find(node.op_);
    if (it != ops.end()) { it->second++; }
    else { ops.insert(std::make_pair(node.op_, 1)); }
  }
  for (auto &op : ops) {
    std::cout << autoda::ops::fullname(op.first) << ":" << op.second << std::endl;
  }
  std::cout << "\n";

  unsigned class_0 = 0;
  unsigned class_1 = 1;
  auto &ds_0 = autoda::cifar2::dataset(class_0);
  auto &ds_1 = autoda::cifar2::dataset(class_1);
  auto &ds_idxes_0 = autoda::cifar2::dataset_idxes_0(class_0, class_1);
  auto &ds_idxes_1 = autoda::cifar2::dataset_idxes_1(class_0, class_1);

  fstd::dynarray<std::unique_ptr<autoda::TacVM>> vms(ds_idxes_0.size());
  Eigen::Array<float, Eigen::Dynamic, 1> dists(ds_idxes_0.size());
  for (size_t i = 0; i < ds_idxes_0.size(); i++) {
    auto &vm = vms[i];
    vm = compiled_boundary->alloc_vm(autoda::cifar2::x_dim);
    boundary->init_.run_S(*vm);
    vm->V_.col(0) = ds_0.col(ds_idxes_0.at(i));
    vm->V_.col(1) = ds_1.col(ds_idxes_1.at(i % ds_idxes_1.size()));
    dists(i) = (vm->V_.col(0) - vm->V_.col(1)).matrix().norm();
  }

  using xs_type = Eigen::Array<float, autoda::cifar2::x_dim, Eigen::Dynamic>;
  autoda::TFTensor<float> xs({(int64_t) ds_idxes_0.size(), autoda::cifar2::x_dim});
  Eigen::Map<xs_type> xs_eigen(xs.get_data(), autoda::cifar2::x_dim, ds_idxes_0.size());
  using labels_type = Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>;
  /* verify the starting points */ {
    for (size_t i = 0; i < ds_idxes_0.size(); i++) {
      xs_eigen.col(i) = vms[i]->V_.col(1);
    }
    auto rs = autoda::cifar2::run(class_0, class_1, xs.raw_);
    Eigen::Map<labels_type> labels(std::get<0>(rs).get_data(), ds_idxes_0.size(), 1);
    for (long i = 0; i < labels.size(); i++) {
      autoda::check(labels(i) == 1);
    }
  }

  Eigen::Array<float, Eigen::Dynamic, 1> src_steps(ds_idxes_0.size(), 1);
  constexpr float decay_factor = 0.95;
  constexpr float l = 0.5;
  constexpr float h = 1.5;
  constexpr float bar_p = 0.25;
  fstd::dynarray<float> succ_rates(ds_idxes_0.size());
  for (auto &succ_rate : succ_rates) succ_rate = bar_p;
  autoda::Stopwatch stopwatch;
  for (size_t step = 0; step < 20000; step++) {
    size_t cpu_time;
    size_t gpu_time;
    stopwatch.start();
    for (size_t i = 0; i < ds_idxes_0.size(); i++) {
      auto &vm = vms[i];
      autoda_fill_randn(vm->V_.col(2));
      compiled_boundary->attack_.tac_.execute(*vm);
      auto V_output_idx = compiled_boundary->attack_.V_output_idx_;
      vm->V_.col(V_output_idx) = autoda_clip(vm->V_.col(V_output_idx));
      float *x_adv = vm->V_.col(V_output_idx).data();
      memcpy(xs_eigen.col(i).data(), x_adv, sizeof(float) * xs_eigen.rows());
    }
    cpu_time = stopwatch.stop();
    stopwatch.start();
    auto rs = autoda::cifar2::run(class_0, class_1, xs.raw_);
    gpu_time = stopwatch.stop();
    stopwatch.start();
    Eigen::Map<labels_type> labels(std::get<0>(rs).get_data(), xs_eigen.cols(), 1);
    for (size_t i = 0; i < ds_idxes_0.size(); i++) {
      auto &vm = vms[i];
      auto &succ_rate = succ_rates[i];
      uint8_t label = labels(i);
      succ_rate *= decay_factor;
      if (label == 1) {
        succ_rate += 1 - decay_factor;
        vm->V_.col(1) = xs_eigen.col(i);
        dists(i) = (vm->V_.col(0) - vm->V_.col(1)).matrix().norm();
      }
      float ratio;
      if (succ_rate >= bar_p) {
        ratio = ((h - 1.0) / (1.0 - bar_p)) * (succ_rate - bar_p) + 1.0;
      } else {
        ratio = ((l - 1.0) / (0.0 - bar_p)) * (succ_rate - bar_p) + 1.0;
      }
      vm->S_(0) *= powf(ratio, 0.1);
      src_steps(i) = vm->S_(0);
    }
    cpu_time += stopwatch.stop();
    float succ = (dists < 1).cast<float>().mean();
    printf(
        "step=%5lu  dist_mean=%10.6f  src_step=%10.6f,%10.6f,%10.6f  "
        "succ=%5.3f  count=%4d  cpu_time=%.2fms  gpu_time=%.2fms\n",
        step, dists.mean(), src_steps.minCoeff(), src_steps.mean(), src_steps.maxCoeff(),
        succ, labels.cast<unsigned>().sum(), cpu_time / 1000.0, gpu_time / 1000.0);
  }

  return 0;
}
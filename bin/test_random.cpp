#include "models.hpp"
#include "random.hpp"
#include "tac.hpp"

using namespace autoda;
using model = cifar2;
unsigned class_0 = 0;
unsigned class_1 = 1;

static constexpr float SUCC_RATE = 0.2;
static constexpr float SUCC_RATE_DECAY_FACTOR = 0.95;
static constexpr float TUNE_LO = 0.5;
static constexpr float TUNE_HI = 1.5;
static constexpr int N = 1;

int main(int argc, char *argv[]) {
  if (argc != 2 || chdir(argv[1]) < 0) {
    perror("chdir()");
    std::exit(-1);
  }
  autoda::cifar2::initialize();

  TacProgram program{};
  program.emit(ops::VV_V_SUB, 0, 1, 3);
  program.emit(ops::V_S_NORM, 3, 1);
  program.emit(ops::VS_V_DIV, 3, 1, 3);
  program.emit(ops::VS_V_MUL, 2, 0, 4);
  program.emit(ops::VV_V_ADD, 4, 4, 4);
  program.emit(ops::VV_V_ADD, 4, 3, 3);
  program.emit(ops::VV_S_DOT, 4, 3, 2);
  program.emit(ops::VV_V_SUB, 3, 4, 3);
  program.emit(ops::VS_V_MUL, 3, 2, 3);
  program.emit(ops::VS_V_MUL, 3, 1, 3);
  program.emit(ops::VS_V_MUL, 4, 1, 4);
  program.emit(ops::VV_V_SUB, 1, 4, 4);
  program.emit(ops::VV_V_ADD, 3, 4, 3);
  uint8_t output_V_idx = 3;

  fstd::dynarray<float> orig_dists(N);
  fstd::dynarray<float> dists(N);
  fstd::dynarray<float> succ_rates(N, SUCC_RATE);
  std::vector<TacVM> vms{};
  for (size_t i = 0; i < N; i++) {
    vms.emplace_back(cifar2::x_dim, 3, 5);
    auto &vm = vms[i];
    vm.S_(0) = 0.01;
    memcpy(vm.V_.col(0).data(), model::load_dataset<0>(class_0, class_1, i),
           sizeof(float) * model::x_dim);
    memcpy(vm.V_.col(1).data(), model::load_dataset<1>(class_0, class_1, i),
           sizeof(float) * model::x_dim);
    dists[i] = autoda_norm(vm.V_.col(0) - vm.V_.col(1));
    orig_dists[i] = dists[i];
  }

  using xs_type = Eigen::Array<float, autoda::cifar2::x_dim, Eigen::Dynamic>;
  autoda::TFTensor<float> xs({N, model::x_dim});
  Eigen::Map<xs_type> xs_eigen(xs.get_data(), model::x_dim, N);
  std::normal_distribution<float> distri(0, 1);

  for (size_t step = 0; step < 5000; step++) {
    for (size_t i = 0; i < N; i++) {
      auto &vm = vms[i];
      autoda_fill_randn(vm.V_.col(2));
      std::cout << vm.V_.col(2).mean() << std::endl;
      program.execute(vm);
      vm.V_.col(output_V_idx) = autoda_clip(vm.V_.col(output_V_idx));
      auto noise = vm.V_.col(output_V_idx) - vm.V_.col(0);
      float new_dist = autoda_norm(noise);
      if (new_dist > dists[i]) {
        std::cout << "failed" << std::endl;
        vm.V_.col(output_V_idx) = vm.V_.col(0) + (dists[i] / new_dist) * noise;
      }
      xs_eigen.col(i) = vm.V_.col(output_V_idx);
    }
    auto rs = model::run(class_0, class_1, xs.raw_);
    auto labels = std::get<0>(rs).get_data();
    for (size_t i = 0; i < N; i++) {
      auto &succ_rate = succ_rates[i];
      auto &vm = vms[i];
      succ_rate *= SUCC_RATE_DECAY_FACTOR;
      if (labels[i] == 1u) {
        succ_rate += 1 - SUCC_RATE_DECAY_FACTOR;
        vm.V_.col(1) = vm.V_.col(output_V_idx);
        dists[i] = autoda_norm(vm.V_.col(1) - vm.V_.col(0));
      }
      float ratio;
      if (succ_rate >= SUCC_RATE) {
        ratio = ((TUNE_HI - 1.0) / (1.0 - SUCC_RATE)) * (succ_rate - SUCC_RATE) + 1.0;
      } else {
        ratio = ((TUNE_LO - 1.0) / (0.0 - SUCC_RATE)) * (succ_rate - SUCC_RATE) + 1.0;
      }
      vm.S_(0) *= powf(ratio, 1 - SUCC_RATE_DECAY_FACTOR);
    }
    float s = 0.0;
    for (size_t i = 0; i < N; i++) { s += dists[i] / orig_dists[i]; }
    s /= N;
    std::cout << step << " " << s << std::endl;
    for (size_t i = 0; i < N; i++) { std::cout << vms[i].S_(0) << " "; }
    std::cout << std::endl;
    for (size_t i = 0; i < 5; i++) {
      std::cout << vms[0].V_.col(1)(i) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}

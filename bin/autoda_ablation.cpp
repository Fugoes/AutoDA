#include "config.hpp"
#include "models.hpp"
#include "mpmc_queue.hpp"
#include "limited_random_alg.hpp"
#include "output.hpp"

using namespace autoda;

using model = cifar2;
static unsigned class_0 = 0;
static unsigned class_1 = 1;

std::shared_ptr<BoundedMPMCQueue<std::vector<std::unique_ptr<LimitedRandomAlg>>>> q_ptr{nullptr};

std::atomic_bool shutdown_flag_{false};
void shutdown() { shutdown_flag_.store(true); }

size_t target_count_{0};
std::atomic_size_t counter_{0};

#define for_cpu_batch(_i) \
  for (size_t _i = 0; _i < P::cpu_batch_size(); _i++)

struct Stat {
  std::atomic_size_t failed_generate_attack_{0};
  std::atomic_size_t failed_inputs_check_{0};
  std::atomic_size_t failed_dist_check_{0};
  std::atomic_size_t succ_{0};

  void update(size_t failed_generate_attack, size_t failed_inputs_check, size_t failed_dist_check,
              size_t succ) {
    failed_generate_attack_.fetch_add(failed_generate_attack, std::memory_order_seq_cst);
    failed_inputs_check_.fetch_add(failed_inputs_check, std::memory_order_seq_cst);
    failed_dist_check_.fetch_add(failed_dist_check, std::memory_order_seq_cst);
    succ_.fetch_add(succ, std::memory_order_seq_cst);
  }
  inline std::tuple<size_t, size_t, size_t, size_t> get() {
    return {
        failed_generate_attack_.load(std::memory_order_acquire),
        failed_inputs_check_.load(std::memory_order_acquire),
        failed_dist_check_.load(std::memory_order_acquire),
        succ_.load(std::memory_order_acquire)
    };
  }
};

struct AblationCPUTask : public CPUTask<model::x_dim> {
  static constexpr size_t ITER = 100;
  static constexpr size_t N = 10;

  using P = CPUTask<model::x_dim>;

  size_t step_{1};
  std::vector<std::unique_ptr<LimitedRandomAlg>> algs_;
  fstd::dynarray<std::unique_ptr<TacVM>> vms_;
  fstd::dynarray<float> dists_;
  fstd::dynarray<float> orig_dists_;

  AblationCPUTask(size_t cpu_batch_size,
                  std::vector<std::unique_ptr<LimitedRandomAlg>> &&algs)
      : P(cpu_batch_size), algs_(std::move(algs)),
        vms_(P::cpu_batch_size()), dists_(P::cpu_batch_size()), orig_dists_(P::cpu_batch_size()) {
    for_cpu_batch(i) {
      auto j = i % N;
      auto &alg = algs_[i / N];
      auto &vm = vms_[i];
      uint8_t S_slots = alg->compiled_attack_->tac_.S_slots_;
      uint8_t V_slots = alg->compiled_attack_->tac_.V_slots_;
      vm = std::make_unique<TacVM>(model::x_dim, S_slots, V_slots);
      alg->S_init_.run(*vm);
      memcpy(vm->V_.col(0).data(), model::load_dataset<0>(class_0, class_1, j),
             sizeof(float) * model::x_dim);
      memcpy(vm->V_.col(1).data(), model::load_dataset<1>(class_0, class_1, j),
             sizeof(float) * model::x_dim);
      float dist = autoda_norm(vm->V_.col(0) - vm->V_.col(1));
      orig_dists_[i] = dist;
      dists_[i] = dist;
      attack(i);
    }
  }

  ~AblationCPUTask() override {}

  inline void attack(size_t i) {
    auto &alg = algs_[i / N];
    auto &vm = vms_[i];
    auto dist = dists_[i];
    auto V_output_idx = alg->compiled_attack_->V_output_idx_;
    // generate noise
    autoda_fill_randn(vm->V_.col(2));
    // run attack
    alg->compiled_attack_->tac_.execute(*vm);
    // clip
    vm->V_.col(V_output_idx) = autoda_clip(vm->V_.col(V_output_idx));
    // projection if needed
    auto noise = vm->V_.col(V_output_idx) - vm->V_.col(0);
    float new_dist = autoda_norm(noise);
    if (new_dist > dist) {
      vm->V_.col(V_output_idx) = vm->V_.col(0) + (dist / new_dist) * noise;
    }
    // set xs
    P::xs_[i] = vm->V_.col(V_output_idx).data();
  }

  inline void learn(size_t i) {
    auto &alg = algs_[i / N];
    auto &vm = vms_[i];
    auto &dist = dists_[i];
    auto V_output_idx = alg->compiled_attack_->V_output_idx_;
    if (P::get_label(i) == 1) {
      vm->V_.col(1) = vm->V_.col(V_output_idx);
      dist = autoda_norm(vm->V_.col(1) - vm->V_.col(0));
    }
  }

  bool poll(Executor<model::x_dim> &executor) override;
};

bool AblationCPUTask::poll(Executor<model::x_dim> &executor) {
  step_++;
  if (step_ < ITER) { // one iteration
    for_cpu_batch(i) {
      learn(i);
      attack(i);
    }
    return true;
  } else if (step_ == ITER) { // do an extra validation
    for_cpu_batch(i) {
      learn(i);
      P::xs_[i] = vms_[i]->V_.col(1).data();
    }
    return true;
  } else {
    size_t succ_count = 0;
    for_cpu_batch(i) { succ_count += P::get_label(i); }
    check(succ_count == P::cpu_batch_size());

    std::stringstream ss{};
    ss.precision(std::numeric_limits<float>::max_digits10);
    for (size_t k = 0; k < P::cpu_batch_size() / N; k++) {
      float ratio = 0.0;
      for (size_t j = 0; j < N; j++) {
        size_t i = k * N + j;
        ratio += dists_[i] / orig_dists_[i];
      }
      ratio /= N;
      ss << "ratios_mean=" << ratio << "\n";
    }
    output::append(ss);
    output::flush();
    size_t c = P::cpu_batch_size() / N;
    if (counter_.fetch_add(c, std::memory_order_seq_cst) + c >= target_count_) {
      shutdown();
      std::stringstream ss{};
      ss << "exit\n";
      output::append(ss);
      output::flush();
      std::abort();
    } else {
      auto task = q_ptr->dequeue();
      executor.submit(new AblationCPUTask(P::cpu_batch_size(), std::move(task)));
    }
    return false;
  }
}

// METHOD:
// 0: naive
// 1: naive + predefined operations
// 2: compat + predefined operations
// 3: compat
// TRICKS:
// bit-0: inputs-check
// bit-1: dist-test
template<unsigned METHOD, uint8_t TRICKS>
void generate_loop(LimitedRandomAlgCfg cfg, Stat &stat, size_t cpu_batch_size,
                   BoundedMPMCQueue<std::vector<std::unique_ptr<LimitedRandomAlg>>> &queue) {
  static_assert(METHOD == 0 || METHOD == 1 || METHOD == 2 || METHOD == 3);

  DistValidator validator(10, model::x_dim, cfg.dag_max_len_, cfg.dag_max_len_);

  constexpr bool do_inputs_check = (TRICKS & 0b1) != 0;
  constexpr bool do_dist_check = (TRICKS & 0b10) != 0;

  while (true) {
    if constexpr (do_dist_check) {
      for (size_t k = 0; k < validator.vms_.size(); k++) {
        auto &vm = validator.vms_[k];
        memcpy(vm.V_.col(0).data(), model::random_load_dataset<0>(0, 1),
               sizeof(float) * model::x_dim);
      }
      validator.refresh();
    }
    size_t failed_generate_attack = 0, failed_inputs_check = 0, failed_dist_check = 0;
    std::vector<std::unique_ptr<LimitedRandomAlg>> algs{};
    while (true) {
      auto alg = new LimitedRandomAlg(cfg);
      while (true) {
        if (shutdown_flag_.load()) return;
        /* generate attack */ {
          bool failed_generate_attack_flag;
          if constexpr (METHOD == 0) {
            failed_generate_attack_flag = alg->generate_attack_naive(cfg);
          } else if constexpr (METHOD == 1) {
            failed_generate_attack_flag = alg->generate_attack_simple(cfg);
          } else if constexpr (METHOD == 2) {
            failed_generate_attack_flag = alg->generate_attack_compat(cfg);
          } else if constexpr (METHOD == 3) {
            failed_generate_attack_flag = alg->generate_attack_compact_wo_predefined_operations(cfg);
          }
          if (!failed_generate_attack_flag) {
            failed_generate_attack++;
            goto alg_reset;
          }
        }
        if constexpr (do_inputs_check) {
          if (!alg->inputs_check(cfg)) {
            failed_inputs_check++;
            goto alg_reset;
          }
        }
        alg->generate_init(cfg);
        alg->compile();
        if constexpr (do_dist_check) {
          if (!alg->dist_check(cfg, validator)) {
            failed_dist_check++;
            goto alg_reset;
          }
        }
        break;
        alg_reset:
        alg->reset(cfg);
      }
      algs.emplace_back(alg);
      if (algs.size() == cpu_batch_size / AblationCPUTask::N) break;
    }
    queue.enqueue(std::move(algs));
    stat.update(failed_generate_attack, failed_inputs_check, failed_dist_check,
                cpu_batch_size / AblationCPUTask::N);
  }
}

int main(int argc, char *argv[]) {
  const auto config = AblationConfig::from_args(argc, argv);
  /* initialization */ {
    if (chdir(config.dir_.c_str()) < 0) {
      perror("chdir()");
      std::exit(-1);
    }
    model::initialize();
    output::initialize(config.output_);
    check(config.cpu_batch_size_ % AblationCPUTask::N == 0);
    target_count_ = config.count_;
  }
  size_t cpu_batch_size = config.cpu_batch_size_;
  size_t gpu_batch_size = config.gpu_batch_size_;
  class_0 = config.class_0_;
  class_1 = config.class_1_;

  LimitedRandomAlgCfg cfg;

  /* setup cfg */ {
    uint8_t S_slots = 1;
    uint8_t V_slots = 3;
    cfg.dag_max_len_ = 20 + S_slots + V_slots;
    cfg.dag_ops_list_.push_back(ops::SS_S_ADD);
    cfg.dag_ops_list_.push_back(ops::SS_S_SUB);
    cfg.dag_ops_list_.push_back(ops::SS_S_MUL);
    cfg.dag_ops_list_.push_back(ops::SS_S_DIV);
    cfg.dag_ops_list_.push_back(ops::VV_V_ADD);
    cfg.dag_ops_list_.push_back(ops::VV_V_SUB);
    cfg.dag_ops_list_.push_back(ops::VS_V_MUL);
    cfg.dag_ops_list_.push_back(ops::VS_V_DIV);
    cfg.dag_ops_list_.push_back(ops::VV_S_DOT);
    cfg.dag_ops_list_.push_back(ops::V_S_NORM);
    cfg.S_slots_ = S_slots;
    cfg.allowed_S_values_.push_back(0.01);
  }

  /* dump config */ {
    std::stringstream ss{};
    ss << "method=" << config.method_ << "\n"
       << "count=" << config.count_ << "\n"
       << "dag_max_len=" << cfg.dag_max_len_ << "\n";
    for (size_t i = 0; i < cfg.dag_ops_list_.size(); i++) {
      uint8_t op = cfg.dag_ops_list_[i];
      ss << "dag_ops_list[" << i << "]=" << ops::fullname(op) << "\n";
    }
    ss << "S_slots=" << (int) cfg.S_slots_ << "\n";
    for (size_t i = 0; i < cfg.allowed_S_values_.size(); i++) {
      ss << "allowed_S_values[" << i << "]=" << cfg.allowed_S_values_[i] << "\n";
    }
    ss << "cpu_batch_size=" << config.cpu_batch_size_ << "\n";
    ss << "gpu_batch_size=" << config.gpu_batch_size_ << "\n";
    output::append(ss);
    output::flush();
  }

  q_ptr = std::make_shared<BoundedMPMCQueue<std::vector<std::unique_ptr<LimitedRandomAlg>>>>(
      config.gen_threads_ * 4
  );
  Stat stat{};

  std::thread stats([&stat]() {
    auto begin = std::chrono::system_clock::now();
    while (!shutdown_flag_.load()) {
      sleep(1);
      auto[failed_generate_attack, failed_inputs_check, failed_dist_check, succ] = stat.get();
      auto N = failed_generate_attack + failed_inputs_check + failed_dist_check + succ;
      auto end = std::chrono::system_clock::now();
      double time = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
      std::cout
          << "failed_generate_attack: " << failed_generate_attack
          << "(" << failed_generate_attack / (N / 100.0) << ")\n"
          << "failed_inputs_check: " << failed_inputs_check
          << "(" << failed_inputs_check / (N / 100.0) << ")\n"
          << "failed_dist_check: " << failed_dist_check
          << "(" << failed_dist_check / (N / 100.0) << ")\n"
          << "succ: " << succ
          << "(" << succ / (N / 100.0) << ") " << succ / time << " alg/s\n"
          << "total: " << N << " " << N / time << " alg/s\n\n";
      std::cout.flush();
    }
  });

  std::vector<std::thread> gen_threads{};
  for (unsigned i = 0; i < config.gen_threads_; i++) {
    if (config.method_ == "base") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<0, 0b00>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "predefined-operations") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<1, 0b00>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "inputs-check") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<1, 0b01>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "dist-test") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<1, 0b11>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "compact") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<2, 0b11>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "except-predefined-operations") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<3, 0b11>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "except-compact") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<1, 0b11>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "except-inputs-check") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<2, 0b10>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "except-dist-test") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<2, 0b01>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "only-predefined-operations") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<1, 0b00>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "only-compact") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<3, 0b00>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "only-inputs-check") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<0, 0b01>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    } else if (config.method_ == "only-dist-test") {
      gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
        generate_loop<0, 0b10>(std::move(cfg), stat, cpu_batch_size, *q_ptr);
      });
    }
  }

  auto executor = std::make_shared<Executor<model::x_dim>>(gpu_batch_size, cpu_batch_size);
  for (size_t i = 0; i < gpu_batch_size * 2 / cpu_batch_size; i++) {
    auto task = q_ptr->dequeue();
    executor->submit(new AblationCPUTask(cpu_batch_size, std::move(task)));
  }

  std::vector<std::thread> cpu_threads{};
  for (size_t i = 0; i < config.threads_; i++) {
    cpu_threads.emplace_back([executor]() {
      while (!shutdown_flag_.load()) {
        if (!executor->cpu_run()) { logi("task done"); }
      }
    });
  }
  executor->gpu_run([class_0 = config.class_0_, class_1 = config.class_1_](TF_Tensor *xs_batch) {
    if (shutdown_flag_.load()) std::exit(0);
    auto rs = autoda::cifar2::run(class_0, class_1, xs_batch);
    return std::move(std::get<0>(rs));
  });

  return 0;
}

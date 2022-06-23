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
void shutdown() { shutdown_flag_.store(true, std::memory_order_release); }

std::atomic_size_t queries_counter{0};

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

struct LRACPUTask : public CPUTask<model::x_dim> {
  static constexpr size_t ITER = 100;

  using P = CPUTask<model::x_dim>;

  size_t step_{1};
  std::vector<std::unique_ptr<LimitedRandomAlg>> best_algs_;
  std::vector<std::unique_ptr<LimitedRandomAlg>> algs_;
  fstd::dynarray<std::unique_ptr<TacVM>> vms_;
  fstd::dynarray<float> dists_;
  float orig_dist_;

  LRACPUTask(size_t cpu_batch_size,
             std::vector<std::unique_ptr<LimitedRandomAlg>> &&algs,
             std::vector<std::unique_ptr<LimitedRandomAlg>> &&best_algs)
      : P(cpu_batch_size), best_algs_(std::move(best_algs)),
        algs_(std::move(algs)), vms_(P::cpu_batch_size()), dists_(P::cpu_batch_size()) {
    auto x = model::random_load_dataset<0>(class_0, class_1);
    auto starting_point = model::random_load_dataset<1>(class_0, class_1);
    for_cpu_batch(i) {
      auto &alg = algs_[i];
      auto &vm = vms_[i];
      uint8_t S_slots = alg->compiled_attack_->tac_.S_slots_;
      uint8_t V_slots = alg->compiled_attack_->tac_.V_slots_;
      vm = std::make_unique<TacVM>(model::x_dim, S_slots, V_slots);
      alg->S_init_.run(*vm);
      memcpy(vm->V_.col(0).data(), x, sizeof(float) * model::x_dim);
      memcpy(vm->V_.col(1).data(), starting_point, sizeof(float) * model::x_dim);
    }
    orig_dist_ = autoda_norm(vms_[0]->V_.col(0) - vms_[0]->V_.col(1));
    for_cpu_batch(i) { dists_[i] = orig_dist_; }
    for_cpu_batch(i) { attack(i); }
  }

  ~LRACPUTask() override {}

  inline void attack(size_t i) {
    auto &alg = algs_[i];
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
    if (new_dist > dist) { vm->V_.col(V_output_idx) = vm->V_.col(0) + (dist / new_dist) * noise; }
    // set xs
    P::xs_[i] = vm->V_.col(V_output_idx).data();
  }

  inline void learn(size_t i) {
    auto &alg = algs_[i];
    auto &vm = vms_[i];
    auto V_output_idx = alg->compiled_attack_->V_output_idx_;
    // if the new x is adversarial, update it to V[1]
    if (P::get_label(i) == 1) {
      vm->V_.col(1) = vm->V_.col(V_output_idx);
      dists_[i] = autoda_norm(vm->V_.col(1) - vm->V_.col(0));
    }
  }

  bool poll(Executor<model::x_dim> &executor) override;
};

struct LRAWithTuneCPUTask : public CPUTask<model::x_dim> {
  static constexpr size_t ITER = 10000;
  static constexpr size_t N = 10;
  static constexpr float SUCC_RATE = 0.25;
  static constexpr float SUCC_RATE_DECAY_FACTOR = 0.95;
  static constexpr float TUNE_LO = 0.5;
  static constexpr float TUNE_HI = 1.5;

  using P = CPUTask<model::x_dim>;

  size_t step_{1};
  std::vector<std::unique_ptr<LimitedRandomAlg>> algs_;
  fstd::dynarray<std::unique_ptr<TacVM>> vms_;
  fstd::dynarray<float> dists_;
  fstd::dynarray<float> orig_dists_;
  fstd::dynarray<float> succ_rates_;

  LRAWithTuneCPUTask(size_t cpu_batch_size,
                     std::vector<std::unique_ptr<LimitedRandomAlg>> &&algs)
      : P(cpu_batch_size),
        algs_(std::move(algs)), vms_(P::cpu_batch_size()),
        dists_(P::cpu_batch_size()), orig_dists_(P::cpu_batch_size()),
        succ_rates_(P::cpu_batch_size(), SUCC_RATE) {
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

  ~LRAWithTuneCPUTask() override {}

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
      // DON'T reset S[0] here
    }
    // set xs
    P::xs_[i] = vm->V_.col(V_output_idx).data();
  }

  inline void learn(size_t i) {
    auto &alg = algs_[i / N];
    auto &vm = vms_[i];
    auto &succ_rate = succ_rates_[i];
    auto V_output_idx = alg->compiled_attack_->V_output_idx_;
    // if the new x is adversarial, update it to V[1]
    succ_rate *= SUCC_RATE_DECAY_FACTOR;
    if (P::get_label(i) == 1) {
      succ_rate += 1 - SUCC_RATE_DECAY_FACTOR;
      vm->V_.col(1) = vm->V_.col(V_output_idx);
      dists_[i] = autoda_norm(vm->V_.col(1) - vm->V_.col(0));
    }
    // tune S[0] up
    float ratio;
    if (succ_rate >= SUCC_RATE) {
      ratio = ((TUNE_HI - 1.0) / (1.0 - SUCC_RATE)) * (succ_rate - SUCC_RATE) + 1.0;
    } else {
      ratio = ((TUNE_LO - 1.0) / (0.0 - SUCC_RATE)) * (succ_rate - SUCC_RATE) + 1.0;
    }
    vm->S_(0) *= powf(ratio, 0.1);
  }

  bool poll(Executor<model::x_dim> &executor) override;
};

bool LRACPUTask::poll(Executor<model::x_dim> &executor) {
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
    size_t count = 0;
    fstd::dynarray<float> ratios(P::cpu_batch_size());
    fstd::dynarray<size_t> idxes(P::cpu_batch_size());
    for_cpu_batch(i) {
      count += P::get_label(i);
      ratios[i] = dists_[i] / orig_dist_;
      idxes[i] = i;
    }
    check(count == P::cpu_batch_size());
    constexpr size_t k = 1;
    std::nth_element(idxes.begin(), idxes.begin() + k, idxes.end(),
                     [&ratios](const size_t a, const size_t b) { return ratios[a] < ratios[b]; });
    for (size_t i = k; i < P::cpu_batch_size(); i++) {
      for (size_t j = 0; j < k; j++) {
        check(ratios[idxes[i]] >= ratios[idxes[j]]);
      }
    }
    std::stringstream ss{};
    for (size_t i = 0; i < k; i++) { ss << " " << ratios[idxes[i]]; }
    logi(ss.str());

    for (size_t i = 0; i < k; i++) { best_algs_.push_back(std::move(algs_[idxes[i]])); }
    check(best_algs_.size() <= P::cpu_batch_size() / 10);
    if (best_algs_.size() == P::cpu_batch_size() / 10) {
      logi("launching a LRAWithTuneCPUTask");
      executor.submit(new LRAWithTuneCPUTask(
          P::cpu_batch_size(), std::move(best_algs_)
      ));
    } else {
      auto task = q_ptr->dequeue();
      executor.submit(new LRACPUTask(
          P::cpu_batch_size(), std::move(task), std::move(best_algs_)
      ));
    }
    return false;
  }
}

bool LRAWithTuneCPUTask::poll(Executor<model::x_dim> &executor) {
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
    size_t count = 0;
    fstd::dynarray<float> ratios(P::cpu_batch_size());
    fstd::dynarray<size_t> idxes(P::cpu_batch_size());
    for_cpu_batch(i) {
      count += P::get_label(i);
      ratios[i] = dists_[i] / orig_dists_[i];
      idxes[i] = i;
    }
    for (size_t i = 0; i < P::cpu_batch_size() / N; i++) {
      float rs = 0.0;
      for (size_t j = 0; j < 10; j++) { rs += ratios[i * N + j]; }
      rs /= N;
      logi("rs=%f", rs);
      std::stringstream ss{};
      ss.precision(std::numeric_limits<float>::max_digits10);
      ss << "rs=" << rs << "\n";
      ss << "s0 = " << algs_[i]->S_init_.values_[0] << "\n";
      algs_[i]->attack_.dag_.display(ss);
      ss << "-----\n";
      algs_[i]->compiled_attack_->tac_.display(ss);
      output::append(ss);
    }
    output::flush();
    auto task = q_ptr->dequeue();
    executor.submit(new LRACPUTask(
        P::cpu_batch_size(), std::move(task),
        std::vector<std::unique_ptr<LimitedRandomAlg>>()
    ));
    return false;
  }
}

void generate_loop(LimitedRandomAlgCfg cfg, Stat &stat, size_t cpu_batch_size,
                   BoundedMPMCQueue<std::vector<std::unique_ptr<LimitedRandomAlg>>> &queue) {
  DistValidator validator(10, model::x_dim, cfg.dag_max_len_, cfg.dag_max_len_);

  while (true) {
    for (size_t k = 0; k < validator.vms_.size(); k++) {
      auto &vm = validator.vms_[k];
      memcpy(vm.V_.col(0).data(), model::random_load_dataset<0>(0, 1), sizeof(float) *
          model::x_dim);
    }
    validator.refresh();
    size_t failed_generate_attack = 0, failed_inputs_check = 0, failed_dist_check = 0;
    std::vector<std::unique_ptr<LimitedRandomAlg>> algs{};
    while (true) {
      auto alg = new LimitedRandomAlg(cfg);
      while (true) {
        if (shutdown_flag_.load(std::memory_order_acquire)) return;
        /* generate attack */ {
          bool failed_generate_attack_flag;
          failed_generate_attack_flag = alg->generate_attack_compat(cfg);
          if (!failed_generate_attack_flag) {
            failed_generate_attack++;
            goto alg_reset;
          }
        }
        if (!alg->inputs_check(cfg)) {
          failed_inputs_check++;
          goto alg_reset;
        }
        alg->generate_init(cfg);
        alg->compile();
        if (!alg->dist_check(cfg, validator)) {
          failed_dist_check++;
          goto alg_reset;
        }
        break;
        alg_reset:
        alg->reset(cfg);
      }
      algs.emplace_back(alg);
      if (algs.size() == cpu_batch_size) break;
    }
    queue.enqueue(std::move(algs));
    stat.update(failed_generate_attack, failed_inputs_check, failed_dist_check, cpu_batch_size);
  }
}

int main(int argc, char *argv[]) {
  const auto config = LRAConfig::from_args(argc, argv);
  /* initialization */ {
    if (chdir(config.dir_.c_str()) < 0) {
      perror("chdir()");
      std::exit(-1);
    }
    model::initialize();
    output::initialize(config.output_);
  }
  size_t cpu_batch_size = config.cpu_batch_size_;
  size_t gpu_batch_size = config.gpu_batch_size_;
  size_t max_queries = config.max_queries_;
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
    ss << "dag_max_len=" << cfg.dag_max_len_ << "\n";
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
    ss << "max_queries=" << config.max_queries_ << "\n";
    ss << "LRACPUTask::ITER="
       << LRACPUTask::ITER << "\n"
       << "LRAWithTuneCPUTask::ITER="
       << LRAWithTuneCPUTask::ITER << "\n"
       << "LRAWithTuneCPUTask::SUCC_RATE="
       << LRAWithTuneCPUTask::SUCC_RATE << "\n"
       << "LRAWithTuneCPUTask::SUCC_RATE_DECAY_FACTOR="
       << LRAWithTuneCPUTask::SUCC_RATE_DECAY_FACTOR << "\n"
       << "LRAWithTuneCPUTask::TUNE_LO="
       << LRAWithTuneCPUTask::TUNE_LO << "\n"
       << "LRAWithTuneCPUTask::TUNE_HI="
       << LRAWithTuneCPUTask::TUNE_HI << "\n";
    output::append(ss);
    output::flush();
  }

  q_ptr = std::make_shared<BoundedMPMCQueue<std::vector<std::unique_ptr<LimitedRandomAlg>>>>(
      config.gen_threads_ * 4
  );
  Stat stat{};

  std::thread stats([&stat]() {
    auto begin = std::chrono::system_clock::now();
    while (!shutdown_flag_.load(std::memory_order_acquire)) {
      sleep(1);
      auto[failed_generate_attack, failed_inputs_check, failed_dist_check, succ] = stat.get();
      auto N = failed_generate_attack + failed_inputs_check + failed_dist_check + succ;
      auto end = std::chrono::system_clock::now();
      double time = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
      size_t queries = queries_counter.load(std::memory_order_acquire);
      std::cout
          << "failed_generate_attack: " << failed_generate_attack
          << "(" << failed_generate_attack / (N / 100.0) << ")\n"
          << "failed_inputs_check: " << failed_inputs_check
          << "(" << failed_inputs_check / (N / 100.0) << ")\n"
          << "failed_dist_check: " << failed_dist_check
          << "(" << failed_dist_check / (N / 100.0) << ")\n"
          << "succ: " << succ
          << "(" << succ / (N / 100.0) << ") " << succ / time << " alg/s\n"
          << "total: " << N << " " << N / time << " alg/s\n"
          << "queries: " << queries << " " << queries / time << " queries/s\n\n";
      std::cout.flush();
    }
  });

  std::vector<std::thread> gen_threads{};
  for (unsigned i = 0; i < config.gen_threads_; i++) {
    gen_threads.emplace_back([cfg, cpu_batch_size, &stat]() {
      generate_loop(std::move(cfg), stat, cpu_batch_size, *q_ptr);
    });
  }

  auto executor = std::make_shared<Executor<model::x_dim>>(gpu_batch_size, cpu_batch_size);
  for (size_t i = 0; i < gpu_batch_size * 2 / cpu_batch_size; i++) {
    auto task = q_ptr->dequeue();
    executor->submit(new LRACPUTask(
        cpu_batch_size, std::move(task),
        std::vector<std::unique_ptr<LimitedRandomAlg>>()
    ));
  }

  std::vector<std::thread> cpu_threads{};
  for (size_t i = 0; i < config.threads_; i++) {
    cpu_threads.emplace_back([executor]() {
      while (!shutdown_flag_.load(std::memory_order_acquire)) {
        if (!executor->cpu_run()) { logi("task done"); }
      }
    });
  }
  executor->gpu_run([class_0 = config.class_0_, class_1 = config.class_1_,
                        gpu_batch_size, max_queries](TF_Tensor *xs_batch) {
    auto queries = queries_counter.fetch_add(gpu_batch_size, std::memory_order_seq_cst);
    if (queries > max_queries) {
      abort();
    }
    if ((queries / gpu_batch_size) % 1000 == 0) {
      std::stringstream ss{};
      ss << "queries=" << queries << "\n";
      output::append(ss);
    }
    auto rs = autoda::cifar2::run(class_0, class_1, xs_batch);
    return std::move(std::get<0>(rs));
  });

  return 0;
}

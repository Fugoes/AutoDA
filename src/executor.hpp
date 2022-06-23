#ifndef AUTODA_EXECUTOR_HPP
#define AUTODA_EXECUTOR_HPP

#include "mpmc_queue.hpp"

namespace autoda {

// All public methods are thread-safe.
// GPU thread:
// 1. Fetch tasks from GPU queue,
// 2. Do the GPU computation,
// 3. Push tasks to CPU queue.
// CPU threads:
// 1. Fetch tasks from CPU queue,
// 2. Do the CPU computation,
// 3. Push tasks to GPU queue.

template<size_t X_DIM>
struct Executor;

template<size_t X_DIM>
struct CPUTask {
  using labels_type = TFTensor<unsigned char>;

  fstd::dynarray<float *> xs_;
  size_t labels_offset_{SIZE_MAX};
  std::shared_ptr<labels_type> labels_{nullptr};

  inline unsigned char get_label(size_t i) { return labels_->data_[labels_offset_ + i]; }
  inline size_t cpu_batch_size() { return xs_.size(); }

  CPUTask(size_t cpu_batch_size) : xs_(cpu_batch_size) {}
  virtual bool poll(Executor<X_DIM> &executor) = 0;
  virtual ~CPUTask() {}
};

template<size_t X_DIM>
struct GPUTask {
  TFTensor<float> xs_batch_;
  fstd::dynarray<CPUTask<X_DIM> *> cpu_tasks_;
  unsigned int empty_slots_;
  std::atomic<unsigned int> done_submit_;

  GPUTask(size_t gpu_batch_size, size_t cpu_batch_size)
      : xs_batch_({(int64_t) gpu_batch_size, X_DIM}),
        cpu_tasks_(gpu_batch_size / cpu_batch_size),
        empty_slots_(gpu_batch_size / cpu_batch_size),
        done_submit_(0) {
  }
};

template<size_t X_DIM>
struct Executor {
  size_t gpu_batch_size_;
  size_t cpu_batch_size_;
  size_t cpu_batch_per_gpu_batch_;

  Executor(size_t gpu_batch_size, size_t cpu_batch_size);
  inline void gpu_run(std::function<TFTensor<unsigned char>(TF_Tensor * )> gpu_computation);
  inline bool cpu_run();
  void submit(CPUTask<X_DIM> *task);

 private:
  MPMCQueue<CPUTask<X_DIM> *> cpu_q_{};
  MPMCQueue<GPUTask<X_DIM> *> gpu_q_{};

  std::mutex mutex_{};
  GPUTask<X_DIM> *current_gpu_task_{nullptr};
};

template<size_t X_DIM>
Executor<X_DIM>::Executor(size_t gpu_batch_size, size_t cpu_batch_size)
    : gpu_batch_size_(gpu_batch_size), cpu_batch_size_(cpu_batch_size),
      cpu_batch_per_gpu_batch_(gpu_batch_size / cpu_batch_size) {
}

template<size_t X_DIM>
void Executor<X_DIM>::gpu_run(std::function<TFTensor<unsigned char>(
    TF_Tensor * )> gpu_computation) {
  for (;;) {
    GPUTask<X_DIM> *gpu_task = gpu_q_.dequeue();
    if (gpu_task == nullptr) { break; }
    using labels_type = TFTensor<unsigned char>;
    auto labels = std::make_shared<labels_type>(std::move(
        gpu_computation(gpu_task->xs_batch_.raw_)
    ));
    for (size_t i = 0; i < gpu_task->cpu_tasks_.size(); i++) {
      CPUTask<X_DIM> *cpu_task = gpu_task->cpu_tasks_[i];
      cpu_task->labels_offset_ = i * cpu_batch_size_;
      cpu_task->labels_ = labels;
    }
    cpu_q_.enqueue(gpu_task->cpu_tasks_);
    delete gpu_task;
  }
}

template<size_t X_DIM>
bool Executor<X_DIM>::cpu_run() {
  CPUTask<X_DIM> *cpu_task = cpu_q_.dequeue();
  if (cpu_task->poll(*this)) {
    submit(cpu_task);
    return true;
  } else {
    delete cpu_task;
    return false;
  }
}

template<size_t X_DIM>
void Executor<X_DIM>::submit(CPUTask<X_DIM> *task) {
  GPUTask<X_DIM> *gpu_task;
  size_t idx;
  /* lock current_gpu_task_ */ {
    std::lock_guard<decltype(mutex_)> lock(mutex_);
    if (current_gpu_task_ == nullptr) { // allocate a new gpu task
      gpu_task = new GPUTask<X_DIM>(gpu_batch_size_, cpu_batch_size_);
      gpu_task->empty_slots_--;
      current_gpu_task_ = gpu_task;
    } else { // use current gpu task
      gpu_task = current_gpu_task_;
      gpu_task->empty_slots_--;
      if (gpu_task->empty_slots_ == 0) { current_gpu_task_ = nullptr; }
    }
    idx = gpu_task->cpu_tasks_.size() - gpu_task->empty_slots_ - 1;
  }
  /* copy xs */ {
    gpu_task->cpu_tasks_[idx] = task;
    for (size_t i = 0; i < task->xs_.size(); i++) {
      float *x = task->xs_[i];
      if (x != nullptr) {
        memcpy(gpu_task->xs_batch_.data_ + X_DIM * (idx * cpu_batch_size_ + i),
               x, sizeof(float) * X_DIM);
      }
    }
  }
  /* maybe enqueue the task */ {
    unsigned int done_submit = gpu_task->done_submit_.fetch_add(1, std::memory_order_seq_cst);
    if (done_submit + 1 == cpu_batch_per_gpu_batch_) { // gpu_task is ready
      gpu_q_.enqueue(std::move(gpu_task));
    }
  }
}

}

#endif //AUTODA_EXECUTOR_HPP

#ifndef AUTODA_MPMC_QUEUE_HPP
#define AUTODA_MPMC_QUEUE_HPP

namespace autoda {

template<typename T>
struct MPMCQueue {
  inline void enqueue(T &&t) {
    bool was_empty;
    /* lock ts_ */ {
      std::lock_guard<decltype(mutex_)> lock(mutex_);
      was_empty = ts_.empty();
      ts_.push_front(std::move(t));
    }
    if (was_empty) { cond_not_empty_.notify_all(); }
  }
  inline void enqueue(const fstd::dynarray<T> &ts) {
    bool was_empty;
    /* lock ts_ */ {
      std::lock_guard<decltype(mutex_)> lock(mutex_);
      was_empty = ts_.empty();
      for (size_t i = 0; i < ts.size(); i++) { ts_.push_front(std::move(ts[i])); }
    }
    if (was_empty) { cond_not_empty_.notify_all(); }
  }
  inline T dequeue() {
    T t;
    /* lock ts_ */ {
      std::unique_lock<decltype(mutex_)> lock(mutex_);
      while (ts_.empty()) { cond_not_empty_.wait(lock); }
      t = std::move(ts_.back());
      ts_.pop_back();
    }
    return t;
  }

 private:
  std::mutex mutex_{};
  std::condition_variable cond_not_empty_{};
  std::deque<T> ts_{};
};

template<typename T>
struct BoundedMPMCQueue {
  explicit BoundedMPMCQueue(size_t max_size) : max_size_(max_size) {}
  inline void enqueue(T &&t) {
    size_t size;
    /* lock ts_ */ {
      std::unique_lock<decltype(mutex_)> lock(mutex_);
      while (true) {
        size = ts_.size();
        if (size < max_size_) break;
        cond_not_full_.wait(lock);
      }
      ts_.push_front(std::move(t));
    }
    if (size == 0) { cond_not_empty_.notify_all(); }
  }
  inline T dequeue() {
    T t;
    size_t size;
    /* lock ts_ */ {
      std::unique_lock<decltype(mutex_)> lock(mutex_);
      while (true) {
        size = ts_.size();
        if (size > 0) break;
        cond_not_empty_.wait(lock);
      }
      t = std::move(ts_.back());
      ts_.pop_back();
    }
    if (size == max_size_) { cond_not_full_.notify_all(); }
    return t;
  }

 private:
  std::mutex mutex_{};
  size_t max_size_;
  std::condition_variable cond_not_empty_{};
  std::condition_variable cond_not_full_{};
  std::deque<T> ts_{};
};

}

#endif //AUTODA_MPMC_QUEUE_HPP

#ifndef AUTODA_MODELS_HPP
#define AUTODA_MODELS_HPP

#include "prelude.hpp"
#include "random.hpp"

namespace autoda {

class cifar2 {
 public:
  static constexpr unsigned x_dim = 32 * 32 * 3;

 private:
  TF_Status *status_{nullptr};
  TF_Session *session_{nullptr};
  TF_Graph *graph_{nullptr};
  TF_Operation **operations_{nullptr};

  Eigen::Matrix<float, x_dim, Eigen::Dynamic> datasets_[10];
  fstd::dynarray<unsigned> datasets_idxes_0_[90];
  fstd::dynarray<unsigned> datasets_idxes_1_[90];

  cifar2() = default;

  static inline cifar2 &instance() noexcept {
    static cifar2 singleton{};
    return singleton;
  }

  static inline unsigned get_i(unsigned class_0, unsigned class_1) {
    unsigned i = class_0 * 9 + class_1;
    if (class_1 > class_0) i -= 1;
    return i;
  }

 public:
  static void initialize();

  using labels_type = TFTensor<unsigned char>;
  using logits_type = TFTensor<float>;
  using probabilities_type = TFTensor<float>;
  static std::tuple<labels_type, logits_type, probabilities_type>
  run(unsigned class_0, unsigned class_1, TF_Tensor *xs);

  ~cifar2();

  static inline const Eigen::Matrix<float, x_dim, Eigen::Dynamic> &dataset(unsigned class_id) {
    return instance().datasets_[class_id];
  }

  static inline const fstd::dynarray<unsigned> &
  dataset_idxes_0(unsigned class_0, unsigned class_1) {
    return instance().datasets_idxes_0_[get_i(class_0, class_1)];
  }

  static inline const fstd::dynarray<unsigned> &
  dataset_idxes_1(unsigned class_0, unsigned class_1) {
    return instance().datasets_idxes_1_[get_i(class_0, class_1)];
  }

  template<unsigned TARGET>
  static const float *
  random_load_dataset(unsigned class_0, unsigned class_1) {
    unsigned lo = 0;
    unsigned hi;
    if constexpr (TARGET == 0) {
      hi = dataset_idxes_0(class_0, class_1).size() - 1;
    } else {
      hi = dataset_idxes_1(class_0, class_1).size() - 1;
    }
    std::uniform_int_distribution<unsigned> dist(lo, hi);
    if constexpr (TARGET == 0) {
      return dataset(class_0).col(dataset_idxes_0(class_0, class_1)[dist(gen)]).data();
    } else {
      return dataset(class_1).col(dataset_idxes_1(class_0, class_1)[dist(gen)]).data();
    }
  }

  template<unsigned TARGET>
  static const float *
  load_dataset(unsigned class_0, unsigned class_1, size_t idx) {
    if constexpr (TARGET == 0) {
      return dataset(class_0).col(dataset_idxes_0(class_0, class_1)[idx]).data();
    } else {
      return dataset(class_1).col(dataset_idxes_1(class_0, class_1)[idx]).data();
    }
  }
};

}

#endif //AUTODA_MODELS_HPP

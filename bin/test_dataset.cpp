#include "models.hpp"

int main(int argc, char *argv[]) {
  if (argc != 2 || chdir(argv[1]) < 0) {
    perror("chdir()");
    std::exit(-1);
  }
  autoda::cifar2::initialize();

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (i != j) {
        auto &idxes_0 = autoda::cifar2::dataset_idxes_0(i, j);
        auto &idxes_1 = autoda::cifar2::dataset_idxes_1(i, j);

        auto xs_len = idxes_0.size() + idxes_1.size();
        autoda::TFTensor<float> xs({(int64_t) xs_len, autoda::cifar2::x_dim});
        float *xs_data = xs.get_data();

        size_t k;
        for (k = 0; k < idxes_0.size(); k++) {
          memcpy(xs_data + k * autoda::cifar2::x_dim,
                 autoda::cifar2::dataset(i).col(idxes_0.at(k)).data(),
                 sizeof(float) * autoda::cifar2::x_dim);
        }
        for (; k < xs_len; k++) {
          memcpy(xs_data + k * autoda::cifar2::x_dim,
                 autoda::cifar2::dataset(j).col(idxes_1.at(k - idxes_0.size())).data(),
                 sizeof(float) * autoda::cifar2::x_dim);
        }

        printf("testing class_0=%u class_1=%u\n", i, j);

        auto result = autoda::cifar2::run(i, j, xs.raw_);
        unsigned char *labels = std::get<0>(result).get_data();
        for (size_t l = 0; l < xs_len; l++) {
          if (l < idxes_0.size()) {
            autoda::check(labels[l] == 0);
          } else {
            autoda::check(labels[l] == 1);
          }
        }
      }
    }
  }

  return 0;
}
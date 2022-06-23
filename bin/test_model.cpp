#include "config.hpp"
#include "models.hpp"

void test(int class_0, int class_1) {
  auto &ds_0 = autoda::cifar2::dataset(class_0);
  auto &ds_1 = autoda::cifar2::dataset(class_1);
  auto xs_len = ds_0.cols() + ds_1.cols();
  autoda::TFTensor<float> xs({(int64_t) xs_len, autoda::cifar2::x_dim});
  float *xs_data = xs.get_data();
  memcpy(xs_data, ds_0.data(), sizeof(float) * ds_0.size());
  memcpy(xs_data + ds_0.size(), ds_1.data(), sizeof(float) * ds_1.size());
  auto result = autoda::cifar2::run(class_0, class_1, xs.raw_);
  using LabelsMat = Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>;
  unsigned char *labels_ptr = std::get<0>(result).get_data();
  Eigen::Map<LabelsMat> labels_0(labels_ptr, ds_0.cols(), 1);
  Eigen::Map<LabelsMat> labels_1(labels_ptr + ds_0.cols(), ds_1.cols(), 1);
  double acc_0 = 1.0 - labels_0.cast<double>().mean();
  double acc_1 = labels_1.cast<double>().mean();
  printf("class_0=%d  class_1=%d  acc_0=%5.3f  acc_1=%5.3f  acc=%5.3f\n",
         class_0, class_1, acc_0, acc_1, (acc_0 + acc_1) / 2);
}

int main(int argc, char **argv) {
  autoda::TestConfig config = autoda::config_from_args(argc, argv);
  if (chdir(config.dir_.c_str()) < 0) {
    perror("chdir()");
    std::exit(-1);
  }

  autoda::cifar2::initialize();

  for (int class_0 = 0; class_0 < 10; class_0++)
    for (int class_1 = 0; class_1 < 10; class_1++)
      if (class_0 != class_1)
        test(class_0, class_1);
  for (int class_0 = 0; class_0 < 10; class_0++)
    for (int class_1 = 0; class_1 < 10; class_1++)
      if (class_0 != class_1)
        test(class_0, class_1);

  return 0;
}

#include "models.hpp"

#include "model_info.hpp.gen"

#include <H5Cpp.h>

namespace autoda {

void cifar2::initialize() {
  instance().status_ = TF_NewStatus();

  // >>> gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
  // >>> proto = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
  // >>> list(map(hex, proto.SerializeToString()))
  // ['0x32', '0x2', '0x20', '0x1']
  static const char options_pb[] = {0x32, 0x2, 0x20, 0x1};
  static const char *tags = "serve";
  static const char *export_dir = "CIFAR-2";
  TF_SessionOptions *options = TF_NewSessionOptions();
  TF_SetConfig(options, options_pb, sizeof(options_pb), instance().status_);
  check(instance().status_);
  instance().graph_ = TF_NewGraph();
  instance().session_ = TF_LoadSessionFromSavedModel(
      options, nullptr, export_dir, &tags, 1, instance().graph_, nullptr, instance().status_);
  TF_DeleteSessionOptions(options);
  check(instance().status_);

  instance().operations_ = new TF_Operation *[2 * 9 * 10];

  for (const auto &x : MODEL_ENTRIES) {
    unsigned class_0 = std::get<0>(x);
    unsigned class_1 = std::get<1>(x);
    const char *input_oper_name = std::get<2>(x);
    const char *output_oper_name = std::get<3>(x);

    TF_Operation *input_oper = TF_GraphOperationByName(instance().graph_, input_oper_name);
    check(input_oper != nullptr);

    TF_Operation *output_oper = TF_GraphOperationByName(instance().graph_, output_oper_name);
    check(output_oper != nullptr);

    unsigned i = cifar2::get_i(class_0, class_1);
    instance().operations_[2 * i] = input_oper;
    instance().operations_[2 * i + 1] = output_oper;
  }

  /* reading dataset */ {
    static const H5std_string DATASET_FILE_NAME("CIFAR-10.h5");
    H5::H5File file(DATASET_FILE_NAME, H5F_ACC_RDONLY);
    for (unsigned i = 0; i < 10; i++) {
      H5::DataSet dataset = file.openDataSet("test_" + std::to_string(i));
      H5::DataSpace space = dataset.getSpace();
      const unsigned n_points = space.getSimpleExtentNpoints();
      const unsigned n_cols = n_points / cifar2::x_dim;
      check(n_points == n_cols * cifar2::x_dim);
      std::unique_ptr<unsigned char[]> imgs(new unsigned char[n_points]);
      auto tc = dataset.getTypeClass();
      check(tc == H5T_INTEGER);
      auto int_type = dataset.getIntType();
      auto order = int_type.getOrder();
      auto size = int_type.getSize();
      check(order == H5T_ORDER_LE);
      check(size == 1);
      dataset.read(imgs.get(), H5::PredType::NATIVE_UINT8);

      instance().datasets_[i].resize(cifar2::x_dim, n_cols);
      for (unsigned j = 0; j < n_points; j++) {
        instance().datasets_[i](j) = imgs[j] / 255.0;
      }
    }
  }

  /* read clean data indexes for each model */ {
    static const H5std_string CLEAN_FILE_NAME("CIFAR-2-CLEAN.h5");
    H5::H5File file(CLEAN_FILE_NAME, H5F_ACC_RDONLY);
    for (unsigned class_0 = 0; class_0 < 10; class_0++) {
      for (unsigned class_1 = 0; class_1 < 10; class_1++) {
        if (class_0 != class_1) {
          unsigned i = cifar2::get_i(class_0, class_1);
          std::ostringstream os{};
          os << "model_" << class_0 << "_" << class_1 << "_";
          std::string dataset_name = os.str();

          /* read indexes for class 0 */ {
            H5::DataSet dataset = file.openDataSet(dataset_name + "0");
            H5::DataSpace space = dataset.getSpace();
            const unsigned n_points = space.getSimpleExtentNpoints();
            fstd::dynarray<unsigned> idxes_0(n_points);
            auto tc = dataset.getTypeClass();
            check(tc == H5T_INTEGER);
            auto int_type = dataset.getIntType();
            auto order = int_type.getOrder();
            auto size = int_type.getSize();
            check(order == H5T_ORDER_LE);
            check(size == 4);
            dataset.read(idxes_0.data(), H5::PredType::NATIVE_UINT32);
            instance().datasets_idxes_0_[i].swap(idxes_0);
          }

          /* read indexes for class 1 */ {
            H5::DataSet dataset = file.openDataSet(dataset_name + "1");
            H5::DataSpace space = dataset.getSpace();
            const unsigned n_points = space.getSimpleExtentNpoints();
            fstd::dynarray<unsigned> idxes_1(n_points);
            auto tc = dataset.getTypeClass();
            check(tc == H5T_INTEGER);
            auto int_type = dataset.getIntType();
            auto order = int_type.getOrder();
            auto size = int_type.getSize();
            check(order == H5T_ORDER_LE);
            check(size == 4);
            dataset.read(idxes_1.data(), H5::PredType::NATIVE_UINT32);
            instance().datasets_idxes_1_[i].swap(idxes_1);
          }
        }
      }
    }
  }
}

cifar2::~cifar2() {
  TF_CloseSession(session_, status_);
  check(status_);
  TF_DeleteSession(session_, status_);
  check(status_);
  TF_DeleteGraph(graph_);
  TF_DeleteStatus(status_);
  delete[] operations_;
}

std::tuple<cifar2::labels_type, cifar2::logits_type, cifar2::probabilities_type>
cifar2::run(unsigned class_0, unsigned class_1, TF_Tensor *xs) {
  unsigned i = cifar2::get_i(class_0, class_1);
  TF_Operation *input_oper = instance().operations_[2 * i];
  TF_Operation *output_oper = instance().operations_[2 * i + 1];

  TF_Tensor *input_values[1];
  input_values[0] = xs;

  TF_Tensor *output_values[3];

  TF_Output inputs[1];
  inputs[0].oper = input_oper;
  inputs[0].index = 0;

  TF_Output outputs[3];
  outputs[0].oper = output_oper;
  outputs[0].index = 0;
  outputs[1].oper = output_oper;
  outputs[1].index = 1;
  outputs[2].oper = output_oper;
  outputs[2].index = 2;

  TF_SessionRun(
      instance().session_, nullptr,
      // input tensors
      inputs, input_values, 1,
      // output tensors
      outputs, output_values, 3,
      nullptr, 0, nullptr, instance().status_
  );
  check(instance().status_);

  return {output_values[0], output_values[1], output_values[2]};
}

}

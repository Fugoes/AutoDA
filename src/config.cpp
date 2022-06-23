#include "config.hpp"

#include <boost/program_options.hpp>

namespace autoda {

TestConfig config_from_args(int argc, char **argv) {
  boost::program_options::options_description desc{"Allowed options"};
  desc.add_options()
      ("help", "produce help message")
      ("dir", boost::program_options::value<std::string>(), "work directory");
  boost::program_options::variables_map vm{};
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  TestConfig config{};

  if (vm.count("help")) goto help;
  if (!vm.count("dir")) goto help;
  if (!vm.count("threads")) goto help;

  config.dir_ = vm["dir"].as<std::string>();

  return config;

  help:
  std::cout << desc << "\n";
  std::exit(-1);
}

AblationConfig AblationConfig::from_args(int argc, char **argv) {
  boost::program_options::options_description desc{"Allowed options"};
  desc.add_options()
      ("help", "produce help message")
      ("dir", boost::program_options::value<std::string>(),
       "work directory")
      ("threads", boost::program_options::value<unsigned>(),
       "cpu threads for executor")
      ("gen-threads", boost::program_options::value<unsigned>(),
       "cpu threads for generating algorithms")
      ("class-0", boost::program_options::value<unsigned>(),
       "class 0 for evaluation")
      ("class-1", boost::program_options::value<unsigned>(),
       "class 1 for evaluation")
      ("cpu-batch-size", boost::program_options::value<unsigned>(),
       "cpu batch size for executor")
      ("gpu-batch-size", boost::program_options::value<unsigned>(),
       "gpu batch size for executor")
      ("method", boost::program_options::value<std::string>(),
       "method for generating algorithms for GPU evaluation")
      ("count", boost::program_options::value<size_t>(),
       "number of algorithms to evaluate on GPU")
      ("output", boost::program_options::value<std::string>(),
       "output file");

  boost::program_options::variables_map vm{};
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  AblationConfig config{};

  auto help = [&desc]() {
    std::cout << desc << "\n";
    std::exit(-1);
  };

  if (vm.count("help")) help();
  if (!vm.count("dir")) help();
  if (!vm.count("threads")) help();
  if (!vm.count("gen-threads")) help();
  if (!vm.count("class-0")) help();
  if (!vm.count("class-1")) help();
  if (!vm.count("cpu-batch-size")) help();
  if (!vm.count("gpu-batch-size")) help();
  auto cpu_batch_size = vm["cpu-batch-size"].as<unsigned>();
  auto gpu_batch_size = vm["gpu-batch-size"].as<unsigned>();
  if (!(gpu_batch_size % cpu_batch_size == 0)) help();
  if (!vm.count("output")) help();
  if (!vm.count("method")) help();
  if (!vm.count("count")) help();
  auto method = vm["method"].as<std::string>();
  if (method != "base" &&
      method != "predefined-operations" &&
      method != "inputs-check" &&
      method != "dist-test" &&
      method != "compact" &&
      method != "except-predefined-operations" &&
      method != "except-compact" &&
      method != "except-inputs-check" &&
      method != "except-dist-test" &&
      method != "only-predefined-operations" &&
      method != "only-compact" &&
      method != "only-inputs-check" &&
      method != "only-dist-test")
    help();

  config.dir_ = vm["dir"].as<std::string>();
  config.threads_ = vm["threads"].as<unsigned>();
  config.gen_threads_ = vm["gen-threads"].as<unsigned>();
  config.class_0_ = vm["class-0"].as<unsigned>();
  config.class_1_ = vm["class-1"].as<unsigned>();
  config.cpu_batch_size_ = cpu_batch_size;
  config.gpu_batch_size_ = gpu_batch_size;
  config.output_ = vm["output"].as<std::string>();
  config.count_ = vm["count"].as<size_t>();
  config.method_ = method;

  return config;
}

LRAConfig LRAConfig::from_args(int argc, char **argv) {
  boost::program_options::options_description desc{"Allowed options"};
  desc.add_options()
      ("help", "produce help message")
      ("dir", boost::program_options::value<std::string>(),
       "work directory")
      ("threads", boost::program_options::value<unsigned>(),
       "cpu threads for executor")
      ("gen-threads", boost::program_options::value<unsigned>(),
       "cpu threads for generating algorithms")
      ("class-0", boost::program_options::value<unsigned>(),
       "class 0 for evaluation")
      ("class-1", boost::program_options::value<unsigned>(),
       "class 1 for evaluation")
      ("cpu-batch-size", boost::program_options::value<unsigned>(),
       "cpu batch size for executor")
      ("gpu-batch-size", boost::program_options::value<unsigned>(),
       "gpu batch size for executor")
      ("output", boost::program_options::value<std::string>(),
       "output file")
      ("max-queries", boost::program_options::value<size_t>(),
       "max queries to model, 0 means infinity");

  boost::program_options::variables_map vm{};
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  LRAConfig config{};

  auto help = [&desc]() {
    std::cout << desc << "\n";
    std::exit(-1);
  };

  if (vm.count("help")) help();
  if (!vm.count("dir")) help();
  if (!vm.count("threads")) help();
  if (!vm.count("gen-threads")) help();
  if (!vm.count("class-0")) help();
  if (!vm.count("class-1")) help();
  if (!vm.count("cpu-batch-size")) help();
  if (!vm.count("gpu-batch-size")) help();
  auto cpu_batch_size = vm["cpu-batch-size"].as<unsigned>();
  auto gpu_batch_size = vm["gpu-batch-size"].as<unsigned>();
  if (!(gpu_batch_size % cpu_batch_size == 0)) help();
  if (!vm.count("output")) help();
  if (!vm.count("max-queries")) help();

  config.dir_ = vm["dir"].as<std::string>();
  config.threads_ = vm["threads"].as<unsigned>();
  config.gen_threads_ = vm["gen-threads"].as<unsigned>();
  config.class_0_ = vm["class-0"].as<unsigned>();
  config.class_1_ = vm["class-1"].as<unsigned>();
  config.cpu_batch_size_ = cpu_batch_size;
  config.gpu_batch_size_ = gpu_batch_size;
  config.output_ = vm["output"].as<std::string>();
  config.max_queries_ = vm["max-queries"].as<size_t>();
  if (config.max_queries_ == 0) {
    config.max_queries_ = std::numeric_limits<size_t>::max();
  }

  return config;
}

}
# Automated Decision-based Attack

## Compiling and Running AutoDA

This section will guide you through compiling and running our AutoDA.

### Dependencies

1. A Linux distro environment with NVIDIA GPU, we use GTX 1080 Ti GPUs to run our experiments.
2. A C++ compiler with C++17 support. We use `gcc 9.3.0`.
3. `cmake>=3.16`, we use `cmake 3.19.4`.
4. `boost` C++ library, we use `boost 1.74.0`.
5. `hdf5` C++ library, we use `hdf5 1.10.5`.
6. `tensorflow` C library downloaded from https://www.tensorflow.org/install/lang_c, we use
   `tensorflow 2.3.1`.
7. `cudnn 7.6` required by `tensorflow 2.3.1`.
8. `cudatoolkit 10.1` required by `tensorflow 2.3.1`.
9. `eigen` C++ library

You need to install static library, shared library, and header files of these mentioned libraries
when they support it.

### Prepare models and datasets

Please follow instructions in the `prepare_models/README.md` file.

### Compile AutoDA

We use cmake tool to compile AutoDA.

```bash
cd source/code/of/autoda/
mkdir build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j`nproc`
```

### Running AutoDA

In the `build/` directory, the `autoda` binary is for running searching experiments, the
`autoda_ablation` binary is for running ablation study experiments for search method.

Before running experiments, please setup your environment correctly so that all shared libraries
could be found.

The following command will run the searching experiments with 500,000,000 queries.

```bash
./autoda --dir ~/data \
    --threads 12 --gen-threads 20 --class-0 0 --class-1 1 \
    --cpu-batch-size 150 --gpu-batch-size 1500 --max-queries 500000000 \
    --output autoda.log
```

The following command will run the ablation study experiment for base method.

```bash
./autoda_ablation --dir ~/data \
    --threads 8 --gen-threads 20 --class-0 0 --class-1 1 \
    --cpu-batch-size 100 --gpu-batch-size 1000 --method base --count 100000 \
    --output base.log
```

All available methods are `base`, `predefined-operations`, `inputs-check`, `dist-test`, and
`compact`.

The output file includes all results for each experiment.

## Running AutoDA 1st, AutoDA 2nd, and 3 baseline attack methods

They are all put in the `prepare_model/attacker.py` file.

We depend on the following Python packages:
1. `python 3.7`.
2. `pytorch 1.7.1`.
3. `torchvision`.
4. `h5py`.
5. `robustness` from https://github.com/MadryLab/robustness.
6. `filelock`.
7. `tensorflow 2.3.1`.
8. `efficientnet` from https://github.com/qubvel/efficientnet.

```bash
python3 attacker.py --dir ~/data \
    --method autoda_0 --model inception_v3 --offset 200 --count 50 \
    --output autoda_0_inception_v3.h5
```

All supported methods are `autoda_0` (AutoDA 1st), `autoda_1` (AutoDA 2nd), `sign_opt` (Sign-OPT)
, `evolutionary` (Evolutionary), `boundary` (Boundary), `hsja` (HSJA). All supported models are 
`Robustness_nat` (normal ResNet50 on CIFAR-10), `densenet` (normal DenseNet on CIFAR-10), `dla` (normal DLA model on CIFAR-10), `dpn` (normal DPN on CIFAR-10), `Robustness_l2_1_0` 
(adversarial training ResNet50 on CIFAR-10, eps = 1.0), `Madry` (Madry's linf adversarial 
training model, eps = 8/255), `resnet101` (normal ResNet101 on ImageNet), `wide_resnet50_2` (normal WRN50 on ImageNet).

The `Madry` model are from https://github.com/MadryLab/cifar10_challenge, with the snapshot from
https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip. The original model is for TensorFlow 1.x. We export it to the SavedModel format supported by TensorFlow 2.x in the `~/data/Madry/` directory. Other models are from `torchvision`, they should automatically download snapshots for themselves.

The output file records each adversarial example's distance to the original image on each 
iteration.

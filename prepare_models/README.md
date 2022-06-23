The following commands will generate all necessary models and datasets under `~/data` for running 
AutoDA. You need to prepare `ILSVRC2012_val_00000001.JPEG` to `ILSVRC2012_val_00001000.JPEG` 
files under the `~/data/ImageNet` directory.

```bash
# prepare dataset
./dataset.py --dir ~/data
# train base CIFAR-10 model (~/data/efn_base.h5)
./train_base.sh ~/data
# train all CIFAR-2 models based on CIFAR-10 model (~/data/CIFAR-2_*_*/)
./train.sh ~/data
# merge all CIFAR-2 models to a single saved model (~/data/CIFAR-2/)
./merge.py --dir ~/data
# prepare dataset for each model
./clean_dataset.py --dir ~/data
# prepare ImageNet dataset
./imagenet_dataset.py --dir ~/data
```

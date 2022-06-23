#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def generate_dataset(p, d=299):
    if d == 299:
        transformation = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ])
    else:
        transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    xs = []
    for i in range(1, 1001):
        img = Image.open("ImageNet/ILSVRC2012_val_0000{:04d}.JPEG".format(i)).convert('RGB')
        x = (transformation(img) * 255.0).numpy()
        x = x.astype(np.uint8)
        print(x.shape)
        xs.append(x)

    np.save(p, np.stack(xs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    os.chdir(args.dir)

    generate_dataset("ImageNet.npy")
    generate_dataset("ImageNet224.npy", 224)

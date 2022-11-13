# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# See README.md in this directory for instructions on how to use this script.

import re
import argparse
import glob
import os
import PIL.Image
import numpy as np
import sys
import random

import util

import nibabel as nib

OUT_RESOLUTION = 256

# Select train and validation subsets from seismic (these two lists shouldn't overlap)
director = '/Users/catarinapinheiro/Documents/git/noise2noise/datasets/seismic_images/'
files = os.listdir(director)
# Filtering only the files.
files = [f for f in files if os.path.isfile(
    director+'/'+f) and f.endswith(".png")]
random.seed(1000)
random.shuffle(files)

train_perc = 0.6
train_basenames = files[0:int(len(files)*train_perc)]
valid_basenames = files[int(len(files)*train_perc): len(files)]


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


def preprocess_mri(input_files,
                   output_file):
    all_files = sorted(input_files)
    num_images = len(all_files)
    print('Input images: %d' % num_images)
    assert num_images > 0

    resolution = np.asarray(PIL.Image.open(all_files[0]), dtype=np.uint8).shape
    assert len(resolution) == 2  # Expect monochromatic images
    print('Image resolution: %s' % str(resolution))

    crop_size = tuple([((r - 1) | 1) for r in resolution])
    crop_slice = np.s_[:crop_size[0], :crop_size[1]]
    print('Crop size: %s' % str(crop_size))

    img_primal = np.zeros((num_images,) + resolution, dtype=np.uint8)
    img_spectrum = np.zeros((num_images,) + crop_size, dtype=np.complex64)

    print('Processing input files..')
    for i, fn in enumerate(all_files):
        if i % 100 == 0:
            print('%d / %d ..' % (i, num_images))
        img = np.asarray(PIL.Image.open(fn), dtype=np.uint8)
        img_primal[i] = img

        img = img.astype(np.float32) / 255.0 - 0.5
        img = img[crop_slice]
        spec = np.fft.fft2(img).astype(np.complex64)
        spec = fftshift2d(spec)
        img_spectrum[i] = spec

    print('Saving: %s' % output_file)
    util.save_pkl((img_primal, img_spectrum), output_file)


def genpkl(args):
    if args.png_dir is None:
        print('Must specify PNG directory directory with --png-dir')
        sys.exit(1)
    if args.pkl_dir is None:
        print('Must specify PKL output directory directory with --pkl-dir')
        sys.exit(1)

    input_train_files = []
    input_valid_files = []
    for base in train_basenames:
        input_train_files.append(os.path.join(
            args.png_dir, base))
    for base in valid_basenames:
        input_valid_files.append(os.path.join(
            args.png_dir, base))
    print('Num train samples', len(input_train_files))
    print('Num valid samples', len(input_valid_files))
    preprocess_mri(input_files=input_train_files,
                   output_file=os.path.join(args.pkl_dir, 'seismic_train.pkl'))
    preprocess_mri(input_files=input_valid_files,
                   output_file=os.path.join(args.pkl_dir, 'seismic_valid.pkl'))


examples = '''examples:
  # Convert the PNG image files into a Python pickle for use in training:
  python %(prog)s genpkl --png-dir=datasets/seismic-png --pkl-dir=datasets
'''


def main():
    parser = argparse.ArgumentParser(
        description='Convert the seismic dataset into a format suitable for network training',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(help='Sub-commands')

    parser_genpkl = subparsers.add_parser(
        'genpkl', help='PNG to PKL converter (used in training)')
    parser_genpkl.add_argument(
        '--png-dir', help='Directory containing .PNGs saved by with the genpng command')
    parser_genpkl.add_argument(
        '--pkl-dir', help='Where to save the .pkl files for train and valid sets')
    parser_genpkl.set_defaults(func=genpkl)

    args = parser.parse_args()
    if 'func' not in args:
        print('No command given.  Try --help.')
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

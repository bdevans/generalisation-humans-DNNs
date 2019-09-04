# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from functools import wraps
import argparse
import sys
import numpy as np
import math
import functools
import os
import shutil

import tensorflow as tf

from official.resnet import resnet_run_loop

from official.resnet import imagenet_main
from official.resnet import imagenet_preprocessing

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500


import imagenet_16

sys.path.append('./object-recognition-combined/code/')
import image_manipulation

import additional_image_manipulations


def as_perturbation_fn(f):
    @wraps(f)
    def wrapper(image):
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.float32
        perturbed = f(image)
        assert perturbed.dtype in [np.float32, np.float64]
        if perturbed.ndim == 2:
            perturbed = perturbed[..., np.newaxis].repeat(3, axis=2)
        assert image.shape == perturbed.shape
        if perturbed.dtype == np.float64:
            perturbed = perturbed.astype(np.float32)
        return perturbed
    return wrapper


def uniform_noise_multiple(x, rng=np.random.RandomState()):
    levels = np.array([0.00, 0.03, 0.05, 0.10, 0.20, 0.35, 0.60, 0.90])
    width = rng.choice(levels)
    return image_manipulation.uniform_noise(
        x, width=width, contrast_level=0.3, rng=rng)


def salt_and_pepper_noise_multiple(x, rng=np.random.RandomState()):
    levels = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    p = rng.choice(levels)
    return additional_image_manipulations.salt_and_pepper_noise(
        x, p=p, contrast_level=0.3, rng=rng)


def salt_and_pepper_noise_multiple__uniform_noise_multiple(
        x, rng=np.random.RandomState()):
    type_ = rng.randint(2)
    if type_ == 0:
        return salt_and_pepper_noise_multiple(x)
    elif type_ == 1:
        return uniform_noise_multiple(x)
    assert False


def uniform_noise_multiple_partial(x, rng=np.random.RandomState()):
    levels = np.array([0.00, 0.03, 0.05, 0.10, 0.20, 0.35])
    width = rng.choice(levels)
    return image_manipulation.uniform_noise(
        x, width=width, contrast_level=0.3, rng=rng)


def uniform_noise_all(x, rng=np.random.RandomState()):
    width = rng.uniform(0.0, 0.9)
    return image_manipulation.uniform_noise(
        x, width=width, contrast_level=0.3, rng=rng)


def color__uniform_noise_multiple(x, rng=np.random.RandomState()):
    color = rng.uniform() < 0.5
    if color:
        return x
    else:
        return uniform_noise_multiple(x, rng=rng)


def color__grayscale(x, rng=np.random.RandomState()):
    color = rng.uniform() < 0.5
    if color:
        return x
    else:
        return image_manipulation.rgb2gray(x)


def color__grayscale_contrast_multiple__uniform_noise_multiple(
        x, rng=np.random.RandomState()):
    type_ = rng.randint(3)
    if type_ == 0:
        return x
    elif type_ == 1:
        return grayscale_contrast_multiple(x)
    elif type_ == 2:
        return uniform_noise_multiple(x)
    assert False


def grayscale_contrast_multiple(x, rng=np.random.RandomState()):
    levels = np.array([1.0, 0.5, 0.3, 0.15, 0.1, 0.05, 0.03, 0.01])
    level = rng.choice(levels)
    return image_manipulation.grayscale_contrast(
        x, contrast_level=level)


def low_pass_multiple(x, rng=np.random.RandomState()):
    levels = np.array([0, 1, 3, 5, 7, 10, 15, 40])
    level = rng.choice(levels)
    return image_manipulation.low_pass_filter(
        x, std=level)


def high_pass_multiple(x, rng=np.random.RandomState()):
    levels = np.array([np.inf, 3., 1.5, 1., 0.7, 0.55, 0.45, 0.4])
    level = rng.choice(levels)
    if level == np.inf:
        return image_manipulation.rgb2gray(x)
    else:
        return image_manipulation.high_pass_filter(
            x, std=level)


def rotation_multiple(x, rng=np.random.RandomState()):
    angles = np.array([0, 90, 180, 270])
    angle = rng.choice(angles)
    if angle == 0:
        return x
    elif angle == 90:
        return image_manipulation.rotate90(x)
    elif angle == 180:
        return image_manipulation.rotate180(x)
    elif angle == 270:
        return image_manipulation.rotate270(x)


def color__grayscale_contrast__high_pass__low_pass__rotation__uniform_noise(
        x, rng=np.random.RandomState()):
    type_ = rng.randint(6)
    if type_ == 0:
        return x
    elif type_ == 1:
        return grayscale_contrast_multiple(x)
    elif type_ == 2:
        return high_pass_multiple(x)
    elif type_ == 3:
        return low_pass_multiple(x)
    elif type_ == 4:
        return rotation_multiple(x)
    elif type_ == 5:
        return uniform_noise_multiple(x)
    assert False


def color__grayscale_contrast__high_pass__low_pass__rotation__salt_and_pepper_noise(  # noqa: E501
        x, rng=np.random.RandomState()):
    type_ = rng.randint(6)
    if type_ == 0:
        return x
    elif type_ == 1:
        return grayscale_contrast_multiple(x)
    elif type_ == 2:
        return high_pass_multiple(x)
    elif type_ == 3:
        return low_pass_multiple(x)
    elif type_ == 4:
        return rotation_multiple(x)
    elif type_ == 5:
        return salt_and_pepper_noise_multiple(x)
    assert False


def color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__uniform_noise(  # noqa: E501
        x, rng=np.random.RandomState()):
    type_ = rng.randint(7)
    if type_ == 0:
        return x
    elif type_ == 1:
        return grayscale_contrast_multiple(x)
    elif type_ == 2:
        return high_pass_multiple(x)
    elif type_ == 3:
        return low_pass_multiple(x)
    elif type_ == 4:
        return rotation_multiple(x)
    elif type_ == 5:
        return phase_scrambling_multiple(x)
    elif type_ == 6:
        return uniform_noise_multiple(x)
    assert False


def color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__salt_and_pepper_noise(  # noqa: E501
        x, rng=np.random.RandomState()):
    type_ = rng.randint(7)
    if type_ == 0:
        return x
    elif type_ == 1:
        return grayscale_contrast_multiple(x)
    elif type_ == 2:
        return high_pass_multiple(x)
    elif type_ == 3:
        return low_pass_multiple(x)
    elif type_ == 4:
        return rotation_multiple(x)
    elif type_ == 5:
        return phase_scrambling_multiple(x)
    elif type_ == 6:
        return salt_and_pepper_noise_multiple(x)
    assert False


def eidolon_reach_multiple_coherence_1(x, rng=np.random.RandomState()):
    assert False, 'needs eidolon package plus bugfix; but py2 only?'

    reach_levels = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    reach = rng.choice(reach_levels)
    coherence = 1.
    grain = 10.
    return image_manipulation.eidolon_partially_coherent_disarray(
        x, reach, coherence, grain)


def phase_scrambling_multiple(x, rng=np.random.RandomState()):
    levels = np.array([0., 30., 60., 90., 120., 150., 180.])
    level = rng.choice(levels)
    return image_manipulation.phase_scrambling(
        x, width=level, rng=rng)


def placeholder__uniform_noise_multiple(
        x, rng=np.random.RandomState(), placeholder=None):
    type_ = rng.randint(2)
    if type_ == 0:
        return placeholder(x)
    elif type_ == 1:
        return uniform_noise_multiple(x)
    assert False


def parse_record_and_perturb(perturbation, input_dropout, sixteen,
                             raw_record, is_training):
    if sixteen:
        features, label = imagenet_16.parse_record(raw_record, is_training)
        image = features['image']
        weight = features['weight']
    else:
        image, label = imagenet_main.parse_record(raw_record, is_training)

    if perturbation is not None:
        print(f'applying perturbation "{perturbation}"')
        if perturbation == 'grayscale_contrast':
            perturbation_fn = partial(
                image_manipulation.grayscale_contrast,
                contrast_level=0.3)
        elif perturbation == 'uniform_noise':
            perturbation_fn = partial(
                image_manipulation.uniform_noise,
                width=0.2, contrast_level=0.3, rng=np.random.RandomState())
        elif perturbation == 'salt_and_pepper_noise':
            perturbation_fn = partial(
                additional_image_manipulations.salt_and_pepper_noise,
                p=0.5, contrast_level=0.3, rng=np.random.RandomState())
        elif perturbation == 'uniform_noise_multiple':
            perturbation_fn = uniform_noise_multiple
        elif perturbation == 'salt_and_pepper_noise_multiple':
            perturbation_fn = salt_and_pepper_noise_multiple
        elif perturbation == 'uniform_noise_multiple_partial':
            perturbation_fn = uniform_noise_multiple_partial
        elif perturbation == 'uniform_noise_all':
            perturbation_fn = uniform_noise_all
        elif perturbation == 'grayscale':
            perturbation_fn = image_manipulation.rgb2gray
        elif perturbation == 'color__grayscale':
            perturbation_fn = color__grayscale
        elif perturbation == 'color__uniform_noise_multiple':
            perturbation_fn = color__uniform_noise_multiple
        elif perturbation == 'grayscale_contrast_multiple':
            perturbation_fn = grayscale_contrast_multiple
        elif perturbation == 'low_pass_multiple':
            perturbation_fn = low_pass_multiple
        elif perturbation == 'high_pass_multiple':
            perturbation_fn = high_pass_multiple
        elif perturbation == 'rotation_multiple':
            perturbation_fn = rotation_multiple
        elif perturbation == 'phase_scrambling_multiple':
            perturbation_fn = phase_scrambling_multiple
        elif perturbation == 'salt_and_pepper_noise_multiple__uniform_noise_multiple':  # noqa: E501
            perturbation_fn = salt_and_pepper_noise_multiple__uniform_noise_multiple  # noqa: E501
        elif perturbation == 'color__grayscale_contrast_multiple__uniform_noise_multiple':  # noqa: E501
            perturbation_fn = color__grayscale_contrast_multiple__uniform_noise_multiple  # noqa: E501
        elif perturbation == 'color__grayscale_contrast__high_pass__low_pass__rotation__uniform_noise':  # noqa: E501
            perturbation_fn = color__grayscale_contrast__high_pass__low_pass__rotation__uniform_noise  # noqa: E501
        elif perturbation == 'color__grayscale_contrast__high_pass__low_pass__rotation__salt_and_pepper_noise':  # noqa: E501
            perturbation_fn = color__grayscale_contrast__high_pass__low_pass__rotation__salt_and_pepper_noise  # noqa: E501
        elif perturbation == 'color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__uniform_noise':  # noqa: E501
            perturbation_fn = color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__uniform_noise  # noqa: E501
        elif perturbation == 'color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__salt_and_pepper_noise':  # noqa: E501
            perturbation_fn = color__grayscale_contrast__high_pass__low_pass__rotation__phase_scrambling__salt_and_pepper_noise  # noqa: E501
        elif perturbation == 'grayscale__uniform_noise_multiple':
            perturbation_fn = partial(
                placeholder__uniform_noise_multiple,
                placeholder=image_manipulation.rgb2gray)
        elif perturbation == ('grayscale_contrast_multiple'
                              '__uniform_noise_multiple'):
            perturbation_fn = partial(
                placeholder__uniform_noise_multiple,
                placeholder=grayscale_contrast_multiple)
        elif perturbation == ('low_pass_multiple'
                              '__uniform_noise_multiple'):
            perturbation_fn = partial(
                placeholder__uniform_noise_multiple,
                placeholder=low_pass_multiple)
        elif perturbation == ('high_pass_multiple'
                              '__uniform_noise_multiple'):
            perturbation_fn = partial(
                placeholder__uniform_noise_multiple,
                placeholder=high_pass_multiple)
        elif perturbation == ('phase_scrambling_multiple'
                              '__uniform_noise_multiple'):
            perturbation_fn = partial(
                placeholder__uniform_noise_multiple,
                placeholder=phase_scrambling_multiple)
        elif perturbation == ('rotation_multiple'
                              '__uniform_noise_multiple'):
            perturbation_fn = partial(
                placeholder__uniform_noise_multiple,
                placeholder=rotation_multiple)
        else:
            raise ValueError('unknown perturbation')

        perturbation_fn = as_perturbation_fn(perturbation_fn)

        num_channels = 3
        channel_means = imagenet_preprocessing._CHANNEL_MEANS
        neg_channel_means = [-x for x in channel_means]

        image = imagenet_preprocessing._mean_image_subtraction(
            image, neg_channel_means, num_channels)
        image = image / 255.

        shape = image.shape
        image = tf.py_func(
            perturbation_fn,
            [image],
            tf.float32,
            stateful=False,
            name=perturbation)
        image.set_shape(shape)

        image = image * 255.
        image = imagenet_preprocessing._mean_image_subtraction(
            image, channel_means, num_channels)
    else:
        print('finetuning on unperturbed images')

    if input_dropout:
        print('adding input dropout with keep_prob = 0.5')
        image = tf.nn.dropout(image, 0.5, name='input_dropout')

    if sixteen:
        return {'image': image, 'weight': weight}, label
    return image, label


def input_fn(perturbation, input_dropout, sixteen, is_training, data_dir,
             batch_size, num_epochs=1, num_parallel_calls=1, multi_gpu=False):
    """Input function which provides batches for train or eval.
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.
      multi_gpu: Whether this is run multi-GPU. Note that this is only required
        currently to handle the batch leftovers, and can be removed
        when that is handled directly by Estimator.

    Returns:
      A dataset that can be used for iteration.
    """
    filenames = imagenet_main.get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    num_images = is_training and _NUM_IMAGES['train'] or \
        _NUM_IMAGES['validation']

    # Convert to individual records
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if sixteen:
        process_record_dataset_fn = imagenet_16.process_record_dataset
    else:
        process_record_dataset_fn = resnet_run_loop.process_record_dataset

    return process_record_dataset_fn(
        dataset, is_training, batch_size, _SHUFFLE_BUFFER,
        partial(parse_record_and_perturb,
                perturbation, input_dropout, sixteen),
        num_epochs, num_parallel_calls, examples_per_epoch=num_images,
        multi_gpu=multi_gpu)


def imagenet_finetuning_model_fn(
        features, labels, mode, params, boundaries, tempfix):
    """Our model_fn for ResNet to be used with our Estimator."""
    # TODO: right now, the pretrained weights expect inputs to be divided
    # by 255; I reported this and once new weights are available, I
    # should use them and remove this code here;
    # When training from scratch I won't do the downscaling
    # because I already started without it and it should
    # be removed from this later as well;
    # right now that means they are incompatible
    # UPDATE: it doesn't have much off an effect at the training
    # curves because in training mode, the scaling is mostly
    # canceled out by batch norm anyway, and after some training
    # or finetuning, the batch norm exp. averages will have adjusted
    # so that it will also be fine during evaluation;
    # maybe I shouldn't use the fix at all
    # https://github.com/tensorflow/models/issues/3779
    if tempfix:
        features = features / 255
        print('dividing pixel values by 255 as a temporary fix')
    else:
        print('not using temporary fix')
    sys.stdout.flush()

    pretrained_steps = 6085624  # corresponded to batch size 32
    pretrained_steps_per_epoch_new = int(math.ceil(
        _NUM_IMAGES['train'] / params['batch_size']))
    pretrained_epochs_new = int(math.ceil(
        pretrained_steps / pretrained_steps_per_epoch_new))
    print('correcting learning rate boundaries by {} epochs'.format(
        pretrained_epochs_new))

    assert len(boundaries) <= 4
    boundaries = [-1] * (4 - len(boundaries)) + boundaries
    print('epoch boundaries for finetuning: {}'.format(boundaries))
    boundaries = [pretrained_epochs_new + x for x in boundaries]

    decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=boundaries,
        decay_rates=decay_rates)

    result = resnet_run_loop.resnet_model_fn(
        features, labels, mode, imagenet_main.ImagenetModel,
        resnet_size=params['resnet_size'],
        weight_decay=1e-4,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        data_format=params['data_format'],
        version=params['version'],
        loss_filter_fn=None,
        multi_gpu=params['multi_gpu'])

    def add_tensor_summary(name):
        g = tf.get_default_graph()
        t = g.get_tensor_by_name(name)
        assert name[-2:] == ':0'
        tf.summary.tensor_summary(name[:-2], t)

    # added this for debugging / learning rate tweaking
    add_tensor_summary('batch_normalization/moving_mean:0')
    add_tensor_summary('batch_normalization/moving_variance:0')
    add_tensor_summary('batch_normalization_48/moving_mean:0')
    add_tensor_summary('batch_normalization_48/moving_variance:0')

    return result


class FinetuningArgParser(argparse.ArgumentParser):
    """Arguments for configuring and finetuning a Resnet Model.
    """

    def __init__(self, resnet_size_choices=None):
        super(FinetuningArgParser, self).__init__(parents=[
            resnet_run_loop.ResnetArgParser(
                resnet_size_choices=[18, 34, 50, 101, 152, 200])
        ], add_help=False)

        self.add_argument('--perturbation', type=str, default=None)
        self.add_argument('--from_scratch', action='store_true')
        self.add_argument('--lr_boundaries', nargs='*', type=int)
        self.add_argument('--sixteen', action='store_true')
        self.add_argument('--input_dropout', action='store_true')
        self.add_argument('--tempfix', action='store_true')


def main(argv):
    parser = FinetuningArgParser()
    flags = parser.parse_args(args=argv[1:])
    perturbation = flags.perturbation

    print('flags', flags)
    sys.stdout.flush()

    if flags.sixteen:
        assert flags.from_scratch

    if not os.path.exists(flags.model_dir):
        if flags.from_scratch:
            print('creating empty directory {}'.format(flags.model_dir))
            os.mkdir(flags.model_dir)
        else:
            print('copying pretrained model to {}'.format(flags.model_dir))
            shutil.copytree('./pretrained', flags.model_dir)

    if flags.from_scratch:
        # assert tf.train.latest_checkpoint(flags.model_dir) is None
        if flags.sixteen:
            if flags.lr_boundaries is None:
                model_fn = functools.partial(
                    imagenet_16.imagenet_model_fn,
                    boundary_epochs=None)
            else:
                print('=' * 70)
                assert len(flags.lr_boundaries) == 4
                print('training from scratch with custom learning'
                      ' rate boundaries: {}'.format(flags.lr_boundaries))
                model_fn = functools.partial(
                    imagenet_16.imagenet_model_fn,
                    boundary_epochs=flags.lr_boundaries)
        else:
            assert flags.lr_boundaries is None
            model_fn = imagenet_main.imagenet_model_fn
    else:
        assert tf.train.latest_checkpoint(flags.model_dir) is not None
        model_fn = functools.partial(imagenet_finetuning_model_fn,
                                     boundaries=flags.lr_boundaries,
                                     tempfix=flags.tempfix)

    input_function = flags.use_synthetic_data and \
        imagenet_main.get_synth_input_fn() or \
        partial(input_fn, perturbation, flags.input_dropout, flags.sixteen)
    resnet_run_loop.resnet_main(
        flags, model_fn, input_function)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)

#!/usr/bin/env python3
"""
Author: Peter Mawhorter
Sources:
  Base code:
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
  Adornments:
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
  Autoencoding setup:
    https://blog.keras.io/building-autoencoders-in-keras.html

Example of training an autoencoder on trivial inputs to show what the
reconstruction error looks like for novel examples that are simple combinations
of learned pieces and ones that aren't.
"""

#---------#
# Imports #
#---------#

import os
import sys
import glob
import shutil
import pickle
import argparse
import subprocess
import itertools
import warnings
import random
import shlex
import math
import csv
import re

import utils
import impr

import pandas as pd
import pandas.core.dtypes.common as pdt

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
from matplotlib.lines import Line2D

from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats import linregress

from skimage import img_as_float
from skimage.io import imread
from skimage.io import imsave
from skimage.color import convert_colorspace
from skimage.transform import resize

import palettable

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1

#--------------------#
# Default Parameters #
#--------------------#

DEFAULT_PARAMETERS = {
  "options": {
    "quiet": True,
    "seed": 23,
  },

  "input": {
    "common_dir": os.path.join("data", "common"),
    "rare_dir": os.path.join("data", "rare"),

    "image_shape": (32, 16, 3), # target image shape
    "initial_colorspace": "RGB", # colorspace of input images
    "training_colorspace": "HSV", # colorspace to use for training
  },

  "network": {
    # training parameters
    "batch_size": 16,
    "epochs": 100,

    # network layer sizes:
    "conv_sizes": [32, 16],
    "base_flat_size": 512,
    "feature_size": 128,

    # training functions
    "loss_function": "mean_squared_error",

    # network design choices:
    "activation": "relu",
    "sparsen": True, #whether or not to force sparse activation in dense layers
    "regularization_coefficient": 1e-5, #how much l1 norm to add to the loss

    "final_layer_name": "final_layer",
  },

  "output": {
    "directory": "out",
    "history_name": "out-hist-{}.zip",
    "keep_history": 4,
    "example_pool_size": 4,
    "figure_font": "DejaVu Sans",
    "figure_text_size": {
      "title": 20,
      "labels": 18,
      "general": 16
    },
    "variable_marker_base_size": 1,
    "variable_marker_var_size": 19,
  },
}

def load_image(params, fn):
  """
  Loads an image from the given file.
  """
  img = imread(fn)
  img = img_as_float(img)
  img = img[:,:,:3]
  img = resize(
    img,
    params["input"]["image_shape"],
    mode="constant",
    cval=0.0 # black
  )
  img = convert_colorspace(
    img,
    params["input"]["initial_colorspace"],
    params["input"]["training_colorspace"]
  )
  return img

def load_data(params):
  """
  Loads the data from the designated input file(s).
  """
  debug("Loading data...")
  common = []
  dr = params["input"]["common_dir"]
  for fn in os.listdir(dr):
    ffn = os.path.join(dr, fn)
    common.append(load_image(params, ffn))

  rare = []
  dr = params["input"]["rare_dir"]
  for fn in os.listdir(params["input"]["rare_dir"]):
    ffn = os.path.join(dr, fn)
    rare.append(load_image(params, ffn))

  return common, rare

def setup_computation(params):
  """
  sets up a neural network using Keras, and returns an (input, output) pair of
  computation graph nodes.
  """
  input_img = Input(shape=params["input"]["image_shape"])

  x = input_img

  for sz in params["network"]["conv_sizes"]:
    x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

  conv_final = x
  # remember our shape, whatever it is
  # TODO: Not so hacky?
  conv_shape = conv_final._keras_shape[1:]

  x = Flatten()(x)
  flattened_size = x._keras_shape[-1]

  flat_size = params["network"]["base_flat_size"]
  min_flat_size = flat_size

  # Our flat probing layers:
  while flat_size >= params["network"]["feature_size"]:
    reg = None
    if params["network"]["sparsen"]:
      reg = l1(params["network"]["regularization_coefficient"])

    if flat_size // 2 < params["network"]["feature_size"]:
      # this is the final iteration
      x = Dense(
        flat_size,
        activation=params["network"]["activation"],
        activity_regularizer=reg,
        name=params["network"]["final_layer_name"]
      )(x)
    else:
      x = Dense(
        flat_size,
        activation=params["network"]["activation"],
        #activity_regularizer=reg
      )(x)

    min_flat_size = flat_size # remember last value > 1
    flat_size //= 2

  flat_final = x

  flat_size = min_flat_size * 2

  # Return to the original image size:
  while flat_size <= params["network"]["base_flat_size"]:
    x = Dense(
      flat_size,
      activation=params["network"]["activation"],
    )(x)
    flat_size *= 2

    x = Dense(flattened_size, activation=params["network"]["activation"])(x)

    flat_return = x

    x = Reshape(conv_shape)(x)

    for sz in reversed(params["network"]["conv_sizes"]):
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(
        sz,
        (3, 3),
        activation=params["network"]["activation"],
        padding='same'
      )(x)

    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoded = x

    return input_img, decoded

def compile_model(params, input, output):
  """
  Compiles the given model to prepare it for training.
  """
  model = Model(input, output)
  model.compile(
    optimizer='adagrad',
    loss=params["network"]["loss_function"]
  )
  return model

def get_encoding_model(auto_model, params):
  """
  Gets an encoding model from a trained autoencoder by pulling out the final
  layer according to its name.
  """
  return Model(
    inputs=auto_model.input,
    outputs=auto_model.get_layer(params["network"]["final_layer_name"]).output
  )

def create_training_generator(params, data):
  """
  Creates a image data generator for training data using the input image
  directory listed in the given parameters.
  """
  def pairgen():
    nonlocal data
    i = 0
    while True:
      obatch = []
      while len(obatch) < params["network"]["batch_size"]:
        obatch.append(data[i])
        i += 1
        i %= len(data)
      obatch = np.asarray(obatch)
      yield (obatch, obatch)

  return pairgen()


def train_model(params, model, training_gen, n):
  """
  Trains the given model using the given training data generator for the given
  number of epochs. Returns nothing (just alters the weights of the given
  model).
  """
  # Fit the model on the batches returned by the given generator:
  model.fit_generator(
    training_gen,
    steps_per_epoch=1 + int(n / params["network"]["batch_size"]),
    epochs=params["network"]["epochs"]
  )

def rate_image(image, model):
  """
  Returns the reconstruction_error for an individual image, which is just the
  RMSE between the image and its reconstruction (as opposed to the model's loss
  function, which includes an L1 regularization term).
  """
  rec = reconstruct_image(image, model)
  return np.sqrt(np.mean((image - rec)**2))

def compute_reconstruction_errors(params, images, model):
  """
  For each image, computes its reconstruction error under the given
  (autoencoder) model, returning a list of floating point numbers which is the
  same length as the input images list.
  """
  debug("Computing reconstruction errors...")
  result = []
  debug("There are {} example images.".format(len(images)))
  for img in images:
    result.append(rate_image(img, model))

  debug("  ...done computing reconstruction errors.")
  return np.array(result)

def save_image(params, img, filename):
  """
  Saves the given image as the given filename.
  """
  img = convert_colorspace(
    img,
    params["input"]["training_colorspace"],
    params["input"]["initial_colorspace"]
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    imsave(filename, img)

def save_images(params, images, name_template, labels=None):
  """
  Saves the given images into the output directory, putting integer labels into
  the given name template. If a list of labels is given, they will be added as
  text inside the saved images using impr.labeled.
  """
  for i in range(len(images)):
    l = str(labels[i]) if (not (labels is None)) else None
    if l:
      img = impr.labeled(images[i], l, text=(1, 0, 1))
    else:
      img = images[i]
    fn = os.path.join(
      params["output"]["directory"],
      name_template.format("{:03}".format(i))
    )
    save_image(params, img, fn)

def montage_images(params, name_template, label=None):
  """
  Groups images in the output directory according to the given name template,
  after filling in a single '*'. Matching images are montages into a combined
  image, using "montage" for the name slot.
  """
  path = params["output"]["directory"]
  targets = glob.glob(os.path.join(path, name_template.format("*")))
  targets.sort()
  output = os.path.join(path, name_template.format("montage"))
  error=None
  if name_template.endswith("pdf"):
    try:
      subprocess.run([
        "gs",
          "-dBATCH",
          "-dNOPAUSE",
          "-q",
          "-sDEVICE=pdfwrite",
          "-sOutputFile={}".format(output),
      ] + targets
      )
    except Exception as e:
      error = e
  else:
    try:
      subprocess.run([
        "montage",
          "-geometry",
          "+2+2",
      ] + targets + [
          output
      ])
      if not (label is None):
        subprocess.run([
          "mogrify",
            "-label",
            str(label),
            output
        ])
    except Exception as e:
      error = e

  if error:
    print(
      "Error while creating montage:\n{}\n  ...continuing...".format(
        str(error)
      ),
      file=sys.stderr
    )

def collect_montages(params, path, label_dirnames=False):
  """
  Collects all montages in the given directory (and any subdirectories,
  recursively) and groups them into one large combined montage. If
  label_dirnames is given, it labels each image with the name of the directory
  it was taken from.
  """
  montages = []
  for root, dirs, files in os.walk(path):
    for f in files:
      if "montage" in f:
        montages.append(os.path.relpath(os.path.join(root, f)))
  montages.sort()
  with_labels = []
  for m in montages:
    if label_dirnames:
      mdir = os.path.dirname(m)
      mn = mdir.split(os.path.sep)[-1]
    else:
      match = re.search(r"/([^/.]*)\.[^/]*$", m)
      if match:
        mn = match.group(1)
      else:
        mn = m
    with_labels.extend(["-label", mn, m])
  subprocess.run([
    "montage",
      "-geometry",
      "+4+4",
  ] + with_labels + [
      "{}/combined-montage.png".format(path)
  ])


def get_features(params, images, model):
  """
  Given images and a model, returns the features of the images encoded with
  that model.
  """
  encoder = get_encoding_model(model, params)

  features = []
  batch = []
  for img in images:
    batch.append(img)
    if len(batch) >= params["network"]["batch_size"]:
      features.extend(encoder.predict(np.asarray(batch)))
      batch = []

  # handle any leftovers:
  if batch:
    features.extend(encoder.predict(np.asarray(batch)))

  return np.array(features)

def manage_backups(**params):
  debug("Managing output backups...")
  bn = params["output"]["history_name"].format(
    params["output"]["keep_history"] - 1
  )
  if os.path.exists(bn):
    debug(
      "Removing oldest backup '{}' (keeping {}).".format(
        bn,
        params["output"]["keep_history"]
      )
    )
    os.remove(bn)
  for i in range(params["output"]["keep_history"])[-2::-1]:
    bn = params["output"]["history_name"].format(i)
    nbn = params["output"]["history_name"].format(i+1)
    if os.path.exists(bn):
      debug("  ...found.")
      os.rename(bn, nbn)

  if os.path.exists(params["output"]["directory"]):
    bn = params["output"]["history_name"].format(0)
    shutil.make_archive(bn[:-4], 'zip', params["output"]["directory"])
    shutil.rmtree(params["output"]["directory"])

  try:
    os.mkdir(params["output"]["directory"], mode=0o755)
  except FileExistsError:
    pass
  debug("  ...done.")

def reconstruct_image(img, model):
  """
  Given a trained autoencoder model, runs the given image through it and
  returns the reconstructed result.
  """
  img = img.reshape((1,) + img.shape) # pretend it's a batch
  return model.predict(img)[0]


@utils.twolevel_default_params(DEFAULT_PARAMETERS)
def setup_figures(**params):
  """
  Matplotlib setup.
  """
  font_family = params["output"]["figure_font"]
  font_sizes = params["output"]["figure_text_size"]
  mplrc("font", family=font_family, size=font_sizes["general"])
  mplrc(
    "axes",
    titlesize=font_sizes["title"],
    labelsize=font_sizes["labels"]
  )

@utils.twolevel_default_params(DEFAULT_PARAMETERS)
def analyze_dataset(**params):
  """
  Analyzes a dataset by training a convolutional neural network to reconstruct
  the images and then comparing various columns of the dataset against the
  reconstruction error of the network.
  """
  # Seed random number generator (hopefully improve image loading times via
  # disk cache?)
  debug("Random seed is: {}".format(params["options"]["seed"]))
  random.seed(params["options"]["seed"])
  # Backup old output directory and create a new one:
  manage_backups(**params)

  # First load the CSV data:
  common, rare = load_data(params)

  combined = list(itertools.chain(*([common]*20 + [rare] * 3)))

  debug('-'*80)
  debug("Training model...")
  debug("  ...Creating trainable model...")
  inp, comp = setup_computation(params)
  model = compile_model(params, inp, comp)
  debug("  ...done creating model.")
  debug("  Creating training generator...")

  train_gen = create_training_generator(params, combined)
  debug("  ...done creating training generator.")

  debug("  Training model...")
  train_model(params, model, train_gen, len(combined))
  debug("  ...done training model.")

  debug('-'*80)
  debug("Got model:")
  debug(model.summary())
  debug('-'*80)

  reer = compute_reconstruction_errors(params, combined, model)
  common_reer = compute_reconstruction_errors(params, common, model)
  rare_reer = compute_reconstruction_errors(params, rare, model)

  nre = np.min(reer)
  xre = np.max(reer)
  novelty = (reer - nre) / (xre - nre)

  # features
  debug('-'*80)

  features = get_features(params, combined, model)
  common_features = get_features(params, common, model)
  rare_features = get_features(params, rare, model)

  debug('-'*80)
  debug("Saving labeled images...")
  save_images(
    params,
    common,
    "A-common-image-orig-{}.png",
    labels=["{:.4f}".format(reer) for reer in common_reer]
  )
  rec_images = [ reconstruct_image(img, model) for img in common ]
  save_images(
    params,
    rec_images,
    "A-common-image-rec-{}.png",
    labels=["{:.4f}".format(reer) for reer in common_reer]
  )

  save_images(
    params,
    rare,
    "B-rare-image-orig-{}.png",
    labels=["{:.4f}".format(reer) for reer in rare_reer]
  )
  rec_images = [ reconstruct_image(img, model) for img in rare ]
  save_images(
    params,
    rec_images,
    "B-rare-image-rec-{}.png",
    labels=["{:.4f}".format(reer) for reer in rare_reer]
  )

  montage_images(params, "A-common-image-orig-{}.png")
  montage_images(params, "A-common-image-rec-{}.png")
  montage_images(params, "B-rare-image-orig-{}.png")
  montage_images(params, "B-rare-image-rec-{}.png")
  collect_montages(params, params["output"]["directory"])
  debug("  ...done.")

  # Analyze feature variety:
  debug('-'*80)
  debug("Analyzing feature usage...")
  any_on = set()
  for ft in common_features:
    for i, f in enumerate(ft):
      if f != 0:
        any_on.add(i)
  debug("{} active common features:".format(len(any_on)))
  debug(sorted(list(any_on)))

  any_on = set()
  for ft in rare_features:
    for i, f in enumerate(ft):
      if f != 0:
        any_on.add(i)
  debug("{} active rare features:".format(len(any_on)))
  debug(sorted(list(any_on)))
  debug("  ...done analyzing features.")

  debug('-'*80)
  debug("Creating image lineup...")
  by_error = sorted(
    list(
      zip(
        common + rare,
        list(common_reer) + list(rare_reer),
        [20]*len(common) + [3]*len(rare)
      )
    ),
    key=lambda ab: ab[1]
  )
  montage = impr.join(
    [
      impr.frame(
        impr.labeled(
          impr.labeled(
            impr.join([img, reconstruct_image(img, model)], padding=2),
            "{:.3f}".format(err),
            text=(1, 0, 1)
          ),
          str(count),
          text=(1, 0, 1)
        ),
        size=4
      )
        for img, err, count in by_error
    ]
  )
  save_image(
    params,
    montage,
    os.path.join(params["output"]["directory"], "sorted_montage.png")
  )
  debug("  ...done creating image lineup.")

if __name__ == "__main__":
  global debug
  parser = argparse.ArgumentParser(description="Test CNN reconstruction error.")
  parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Suppress progress messages & information."
  )
  parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=23,
    help="Set the random seed."
  )
  options = parser.parse_args()

  debug = utils.get_debug(options.quiet)

  setup_figures()

  analyze_dataset(options=vars(options))

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

Trains an autoencoding network consisting of a CNN which is hooked up to a set
of regular dense layers that decrease by powers of two. The idea is that we'll
be able to use images with minimal autoencoding loss at a given network layer
to represent clusters.
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
import random
import math
import csv
import re

import utils

import numpy as np

from scipy.misc import imread
from scipy.misc import toimage

from scipy.stats import t
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.proportion import proportion_confint

from scikits.bootstrap import ci
from scikits.bootstrap import InstabilityWarning

print("Importing keras...")
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1
print("...done.")

print("Importing matplotlib...")
import matplotlib as mpl
import matplotlib.pyplot as plt
print("...done.")

print("Importing scikit-learn...")
import sklearn
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise
from sklearn.metrics import confusion_matrix
print("...done.")

from skimage.color import convert_colorspace

import palettable

from multiscale import condensed_multiscale
from multiscale import cluster_assignments
from multiscale import typicality

#------------------------------#
# Shims for imported functions #
#------------------------------#

def NovelClustering(points, distances=None, edges=None):
  return cluster_assignments(
    points,
    condensed_multiscale(
      points,
      distances=distances,
      edges=edges,
      quiet=False,
      use_cached_neighbors=False
    )
  )

def simple_proportion(data):
  return np.sum(data) / len(data)

def confidence_interval(
  data,
  distribution="normalish",
  bound=0.05,
  n_samples=10000,
  bootstrap_cutoff=15
):
  n = len(data)
  if distribution == "normalish":
    if n < bootstrap_cutoff:
      return bootstrap_ci(
        data,
        np.average,
        bound=bound,
        n_samples=n_samples
      )
    else:
      return mean_ci(data, bound)
  elif distribution == "binomial":
    if n < bootstrap_cutoff:
      return bootstrap_ci(
        data,
        simple_proportion,
        bound=bound,
        n_samples=n_samples
      )
    else:
      return proportion_confint(
        sum(data),
        len(data),
        alpha=bound,
        method="jeffreys"

      )

def mean_ci(data, bound=0.05):
  """
  Confidence interval for a mean, "by hand". Cribbed from here:

    http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-23-point.html
  and here:

    https://gist.github.com/gcardone/05276131b6dc7232dbaf
  """
  m = np.mean(data)
  s = np.std(data)

  # critical value for Student's t (two-tailed distribution)
  z_crit = t.ppf(1 - (bound/2), len(data)-1)

  margin = z_crit * s / (len(data)**0.5)

  return (m - margin, m + margin)

def bootstrap_ci(
  data,
  statistic,
  bound=0.05,
  n_samples=10000
):
  method = "bca"
  if all(d == data[0] for d in data):
    # zero-variance case, fall back on a simple percentile interval
    method="pi"

  return utils.run_lax(
    [ InstabilityWarning ],
    ci,
    data,
    statistic,
    alpha=bound,
    n_samples=n_samples,
    method=method
  )

#--------------#
# Global Setup #
#--------------#

# Hide TensorFlow info/warnings:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

#---------#
# Globals #
#---------#

# TODO: Get rid of these

ALL_COMPETENCES = [
  "Beginner",
  "Intermediate",
  "Expert",
  "Private",
]

ALL_GENRES = [
  "Action",
  "Adventure",
  "Fighting",
  "Puzzle",
  "Racing",
  "RPG",
  "Simulation",
  "Sports",
  "Shooter",
  "Board Game",
  "Music",
]

#--------------------#
# Default Parameters #
#--------------------#

DEFAULT_PARAMETERS = {
  "options": None,

  "input": {
    #"img_dir": os.path.join("data", "original"),
    "img_dir": os.path.join("data", "mii_flat"),
    "csv_file": os.path.join("data", "csv", "miiverse_profiles.clean.csv"),

    "id_template": re.compile(r"([^_]+)_([^_]+)_.*"), # Matches IDs in filenames
    "image_shape": (48, 48, 3), # target image shape
  },

  "data_processing": {
    "multi_field_separator": '|',
    "field_types": {
      "friends": "integer",
      "following": "integer",
      "followers": "integer",
      "posts": "integer",
      "yeahs": "integer",
      "genres": "multi",
      "country-code": "categorical",
      "competence": "categorical",
    },
    "normalize_fields": [
      "friends", "following", "followers", "posts", "yeahs"
    ],
    "log_transform_fields": [
      "friends", "following", "followers", "posts", "yeahs"
    ],
    "binarize_fields": {
      "friends": { -1: "private", 0: "no-friends" },
      "competence": "auto",
      "country-code": "auto",
    },
    #"filter_on": [ "!private" ],
    "filter_on": [ "!private", "!no-friends" ],
    "subset_size": 10000,
  },

  "network": {
    # training parameters
    "batch_size": 32,
    "percent_per_epoch": 1.0, # how much of the data to feed per epoch
    #"epochs": 200,
    #"epochs": 50,
    "epochs": 10,
    #"epochs": 4,

    # network layer sizes:
    "conv_sizes": [32, 16],
    "base_flat_size": 512,
    "feature_size": 128,

    # training functions
    "ae_loss_function": "mean_squared_error",
    "pr_loss_function": "binary_crossentropy",

    # network design choices:
    "sparsen": True, #whether or not to force sparse activation in dense layers
    "subtract_mean": False, # whether to subtract the mean image before training
    "initial_colorspace": "RGB", # colorspace of input images
    "training_colorspace": "HSV", # colorspace to use for training
    "add_corruption": False, # whether to add corruption when training the AE
    "corruption_factor": 0.1, #how much corruption to add
    "normalize_activation": False, # whether to add normalizing layers or not
    "regularization_coefficient": 1e-5, #how much l1 norm to add to the loss

    # prediction parameters
    "predict_target": ["competence"],

    "final_layer_name": "final_layer",
  },

  "clustering": {
    "metric": "euclidean", # metric for distance measurement

    # typicality parameters:
    "typicality_fraction": 0.008, # percent of points to use for typicality

    #"method": AffinityPropagation,
    #"method": DBSCAN,
    #"method": AgglomerativeClustering,
    "method": NovelClustering,
    #"input": "features",
    "input": "projected",

    "neighborhood_size": 10,

    "dbscan_neighbors": 3,
    "dbscan_percentile": 80,

    "significant_size": 20,
    "viz_size": 10,
  },

  # statistical analysis parameters:
  "analysis": {
    "confidence_baseline": 0.05, # base desired confidence across all tests

    "methods": [
      #"mean_image",
      #"training_examples",
      "reconstructions",
      "reconstruction_error",
      "reconstruction_correlations",
      "typicality_correlations",
      "tSNE",
      #"distance_histograms",
      #"distances",
      #"duplicates",
      "cluster_sizes",
      "cluster_statistics",
      "cluster_samples",
      #"prediction_accuracy",
    ],

    "correlate_with_error": [
      "country-code-US",
      "competence",
      "competence-Beginner",
      "competence-Expert",
      "competence-Intermediate",
      "log-friends",
      "log-following",
      "log-followers",
      "log-posts",
      "log-yeahs",
      #"no-friends", #can't correlate if it's being filtered
      "genres[]",
    ] + [
      "genres[{}]".format(g) for g in ALL_GENRES
    ],

    "analyze_per_cluster": [
      "norm_rating",
      "typicality",
      "country-code",
      "competence-Beginner",
      "competence-Expert",
      "competence-Intermediate",
      "log-friends",
      "log-following",
      "log-followers",
      "log-posts",
      "log-yeahs",
      "genres[]",
    ] + [
      "genres[{}]".format(g) for g in ALL_GENRES
    ],
    "predict_analysis": [ "confusion" ],
    "tsne_subsample": 2000,
  },

  "output": {
    "directory": "out",
    "history_name": "out-hist-{}.zip",
    "keep_history": 4,
    "example_pool_size": 16,
    "large_pool_size": 64,
    "max_cluster_samples": 10000,
    "samples_per_cluster": 16,
  },

  "filenames": {
    "mean_image": "mean-image-{}.png",
    "raw_example": "example-raw-image-{}.png",
    "input_example": "example-input-image-{}.png",
    "output_example": "example-output-image-{}.png",
    "best_image": "A-best-image-{}.png",
    "sampled_image": "A-sampled-image-{}.png",
    "worst_image": "A-worst-image-{}.png",
    "duplicate_image": "duplicate-image-{}.png",

    "correlation_report": "correlation-{}-{}.pdf",
    "histogram": "histogram-{}.pdf",
    "cluster_sizes": "cluster-sizes.pdf",
    "cluster_stats": "cluster-stats-{}.pdf",
    "distances": "distances-{}.pdf",
    "tsne": "tsne-{}-{}x{}.pdf",
    "analysis": "analysis-{}.pdf",
    "cluster_rep": "rep-{}.png",

    "transformed_dir": "transformed",
    "examples_dir": "examples",
    "duplicates_dir": "duplicates",
    "clusters_dir": "clusters",
    "z_dir": "z",
  },
}

def load_data():
  items = {}
  for dp, dn, files in os.walk(params["input"]["img_dir"]):
    for f in files:
      if f.endswith(".jpg") or f.endswith(".png"):
        fbase = os.path.splitext(os.path.basename(f))[0]
        match = params["input"]["id_template"].match(fbase)
        if not match:
          continue
        country = match.group(1)
        id = match.group(2)
        items[id] = os.path.join(dp, f)

  full_items = {}
  values = {"file": "text"}
  types = {"file": "text"}
  legend = None
  print("Reading CSV file...")
  with open(params["input"]["csv_file"], 'r', newline='') as fin:
    reader = csv.reader(fin, dialect="excel")
    legend = next(reader)
    for i, key in enumerate(legend):
      if key in params["data_processing"]["field_types"]:
        types[key] = params["data_processing"]["field_types"]
        if types[key] == "multi":
          values[key] = set()
        elif types[key] == "categorical":
          values[key] = dict()
        else:
          values[key] = types[key]
      else:
        values[key] = "text"
        types[key] = "text"

    for lst in reader:
      if len(lst) != len(legend):
        raise RuntimeWarning(
          "Warning: line(s) with incorrect length {} (expected {}):".format(
            len(lst),
            len(legend)
          )
        )
        print(lst, file=sys.stderr)
        print("Ignoring unparsable line(s).", file=sys.stderr)

      ikey = lst[legend.index("avi-id")]
      if ikey in items:
        record = {}
        for i, val in enumerate(lst):
          col = legend[i]
          if types[col] == "numeric":
            record[col] = float(val)
          elif types[col] == "integer":
            record[col] = int(val)
          elif types[col] == "multi":
            record[col] = val
            for v in val.split(
              params["data_processing"]["multi_field_separator"]
            ):
              values[col].add(v)
          elif types[col] == "categorical":
            if len(values[col]) == 0:
              record[col] = 0
              values[col][val] = 0
            elif val in values[col]:
              record[col] = values[col][val]
            else:
              nv = max(values[col].values()) + 1
              record[col] = nv
              values[col][val] = nv
          else:
            record[col] = val
        record["file"] = items[ikey]
        full_items[ikey] = record

  print("  ...found {} records..".format(len(full_items)))
  print("  ...done.")

  print("Expanding MULTI fields...")
  for col in legend:
    if col in MULTI_FIELDS:
      for v in values[col]:
        values["{}[{}]".format(col, v)] = "boolean"
        types["{}[{}]".format(col, v)] = "boolean"

  lfi = len(full_items)
  for i, (ikey, record) in enumerate(full_items.items()):
    utils.prbar(i / lfi)
    for col in legend:
      if col in MULTI_FIELDS:
        hits = record[col].split(MULTI_FIELDS[col])
        for v in values[col]:
          nfn = "{}[{}]".format(col, v)
          record[nfn] = v in hits

  print("\n  ...done.")

  print("Converting & filtering data...")
  long_items = {"id": []}
  for id in full_items:
    long_items["id"].append(id)
    record = full_items[id]
    for key in record:
      if key not in long_items:
        long_items[key] = []
      long_items[key].append(record[key])

  long_items["values"] = values
  long_items["types"] = types
  long_items["values"]["id"] = "text"
  long_items["types"]["id"] = "text"

  for col in long_items:
    if col in ("values", "types"):
      continue
    if long_items["types"][col] == "numeric":
      long_items[col] = np.asarray(long_items[col], dtype=float)
    elif long_items["types"][col] == "integer":
      long_items[col] = np.asarray(long_items[col], dtype=int)
    elif long_items["types"][col] == "boolean":
      long_items[col] = np.asarray(long_items[col], dtype=bool)
    elif long_items["types"][col] == "categorical":
      long_items[col] = np.asarray(long_items[col], dtype=int)
    else:
      # else use default dtype
      long_items[col] = np.asarray(long_items[col])

  # Normalize some items:
  for col in params["data_processing"]["normalize_fields"]:
    add_norm_column(long_items, col)

  for col in params["data_processing"]["log_transform_fields"]:
    add_log_column(long_items, col)

  # Create binary columns:
  for col in params["data_processing"]["binarize_fields"]:
    if params["data_processing"]["binarize_fields"][col] == "auto":
      if long_items["types"][col] == "categorical":
        for val in long_items["values"][col]:
          vid = long_items["values"][col][val]
          add_binary_column(long_items, col, vid, col + "-" + val)
      else:
        raise RuntimeWarning(
          "Warning: Can't automatically binarize non-categorical column '{}'."
          .format(col)
        )
    else:
      for val in params["data_processing"]["binarize_fields"][col]:
        add_binary_column(
          long_items,
          col,
          val,
          params["data_processing"]["binarize_fields"][col][val]
        )

  precount = len(long_items["id"])

  # Filter data:
  for fil in params["data_processing"]["filter_on"]:
    if fil[0] == "!":
      mask = ~ np.asarray(long_items[fil[1:]], dtype=bool)
    else:
      mask = np.asarray(long_items[fil], dtype=bool)
    for col in long_items:
      if col in ("values", "types"):
        continue
      long_items[col] = long_items[col][mask]

  count = len(long_items["id"])
  print(
    "  Filtered {} items down to {} accepted items...".format(precount, count)
  )

  if count > params["data_processing"]["subset_size"]:
    print(
      "  Subsetting from {} to {} accepted items...".format(
        count,
        params["data_processing"]["subset_size"]
      )
    )
    ilist = list(range(count))
    count = params["data_processing"]["subset_size"]
    random.shuffle(ilist)
    ilist = np.asarray(ilist[:count])
    for col in long_items:
      if col in ("values", "types"):
        continue
      long_items[col] = long_items[col][ilist]

  long_items["count"] = count
  print("  ...done.")

  oldcols = list(long_items["types"].keys())
  for col in oldcols:
    if long_items["types"][col] == "categorical":
      cname = col + "-categorical"
      long_items[cname] = to_categorical(long_items[col])
      long_items["values"][cname] = "one-hot"
      long_items["types"][cname] = "one-hot"

  return long_items

def add_norm_column(items, col):
  nname = col + "-norm"
  items["values"][nname] = "numeric"
  items["types"][nname] = "numeric"
  col_max = np.max(items[col])
  items[nname] = items[col] / col_max

def add_log_column(items, col):
  nname = "log-" + col
  items["values"][nname] = "numeric"
  items["types"][nname] = "numeric"
  #items[nname] = np.log(items[col] + np.min(items[col][items[col]!=0])/2)
  items[nname] = np.log(items[col] + 1)

def add_binary_column(items, col, val, name):
  items["values"][name] = "boolean"
  items["types"][name] = "boolean"
  items[name] = items[col] == val

def ordinal(n):
  if 11 <= n <= 19:
    return str(n) + "th"
  s = str(n)
  last = int(s[-1])
  if 1 <= last <= 3:
    return s + ("st", "nd", "rd")[last-1]
  return s + "th"

def setup_computation(items, mode="autoencoder"):
  # TODO: How does resizing things affect them?
  input_img = Input(shape=params["input"]["image_shape"])

  x = input_img

  for sz in params["network"]["conv_sizes"]:
    x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    if params["network"]["normalize_activation"]:
      x = BatchNormalization()(x)

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
        activation='relu',
        activity_regularizer=reg,
        name=params["network"]["final_layer_name"]
      )(x)
    else:
      x = Dense(
        flat_size,
        activation='relu'
        #activity_regularizer=reg
      )(x)

    if params["network"]["normalize_activation"]:
      x = BatchNormalization()(x)

    # TODO: Smoother layer size reduction?
    min_flat_size = flat_size # remember last value > 1
    flat_size //= 2

  flat_final = x

  flat_size = min_flat_size * 2

  if mode in ["predictor", "dual"]:
    # In predictor mode, we narrow down to the given number of outputs
    outputs = 0
    for t in params["network"]["predict_target"]:
      if len(items[t].shape) > 1:
        outputs += items[t].shape[1]
      else:
        outputs += 1
    predictions = Dense(outputs, activation='relu')(x)
    if params["network"]["normalize_activation"]:
      predictions = BatchNormalization()(predictions)

    if mode == "predictor":
      return input_img, predictions

  if mode in ["autoencoder", "dual"]:
    # In autoencoder mode, we return to the original image size:
    # TODO: construct independent return paths for each probe layer!
    while flat_size <= params["network"]["base_flat_size"]:
      x = Dense(
        flat_size,
        activation='relu',
      )(x)
      # TODO: dropout on the way back up?
      flat_size *= 2

    x = Dense(flattened_size, activation='relu')(x)
    if params["network"]["normalize_activation"]:
      x = BatchNormalization()(x)

    flat_return = x

    x = Reshape(conv_shape)(x)

    for sz in reversed(params["network"]["conv_sizes"]):
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    if params["network"]["normalize_activation"]:
      x = BatchNormalization()(x)

    decoded = x

    if mode == "autoencoder":
      return input_img, decoded
    elif mode =="dual":
      return input_img, (decoded, predictions)

def compile_model(input, output, mode):
  model = Model(input, output)
  # TODO: These choices?
  #model.compile(optimizer='adadelta', loss=LOSS_FUNCTION)
  if mode == "autoencoder":
    model.compile(
      optimizer='adagrad',
      loss=params["network"]["ae_loss_function"]
    )
  else:
    model.compile(
      optimizer='adagrad',
      loss=params["network"]["pr_loss_function"]
    )
  return model

def get_encoding_model(auto_model):
  return Model(
    inputs=auto_model.input,
    outputs=auto_model.get_layer(params["network"]["final_layer_name"]).output
  )

def load_images_into_items(items):
  # TODO: Resize images as they're loaded?
  all_images = []
  for i, filename in enumerate(items["file"]):
    utils.prbar(i / items["count"])
    img = imread(filename)
    img = img[:,:,:3] # throw away alpha channel
    convert_colorspace(
      img,
      params["network"]["initial_colorspace"],
      params["network"]["training_colorspace"]
    )
    img = img / 255
    all_images.append(img)

  print() # done with progress bar

  items["image"] = np.asarray(all_images)
  items["mean_image"] = np.mean(items["image"], axis=0)
  items["image_deviation"] = items["image"] - items["mean_image"]

def create_simple_generator():
  return ImageDataGenerator().flow_from_directory(
    params["input"]["img_dir"],
    target_size=params["input"]["image_shape"][:-1],
    batch_size=1,
    shuffle=False,
    class_mode='sparse' # classes as integers
  )
  
def create_training_generator(items, mode="autoencoder"):
  src = items["image"]
  if params["network"]["subtract_mean"]:
    src = items["image_deviation"]
  if mode == "autoencoder":
    datagen = ImageDataGenerator() # no data augmentation (we eschew generality)

    #train_datagen = datagen.flow_from_directory(
    #  params["input"]["img_dir"],
    #  batch_size=params["network"]["batch_size"],
    #  class_mode='sparse' # classes as integers
    #)
    train_datagen = datagen.flow(
      src,
      src,
      batch_size=params["network"]["batch_size"]
    )

    if params["network"]["add_corruption"]:
      def pairgen():
        while True:
          batch, _ = next(train_datagen)
          # Subtract mean and introduce noise to force better representations:
          for img in batch:
            corrupted = (
              img
            + (
                params["network"]["corruption_factor"]
              * np.random.normal(
                  loc=0.0,
                  scale=1.0,
                  size=img.shape
                )
              )
            )
            yield (corrupted, img)
    else:
      def pairgen():
        while True:
          batch, _ = next(train_datagen)
          # Subtract mean and introduce noise to force better representations:
          for img in batch:
            yield (img, img)

  elif mode == "predictor":
    idx = 0
    # TODO: Shuffle ordering?
    def pairgen():
      nonlocal idx
      while True:
        idx += 1
        idx %= len(src)
        true = []
        for t in params["network"]["predict_target"]:
          if items["types"][t] == "categorical":
            true.extend(items[t + "_categorical"][idx])
          else:
            true.append(items[t][idx])
        yield(src[idx], true)

  else:
    print("Invalid mode '{}'! Aborting.".format(mode))
    exit(1)
  
  def batchgen(pairgen):
    while True:
      batch_in = []
      batch_out = []
      for i in range(params["network"]["batch_size"]):
        inp, outp = next(pairgen)
        batch_in.append(inp)
        batch_out.append(outp)
      yield np.asarray(batch_in), np.asarray(batch_out)

  return batchgen(pairgen())


def train_model(model, training_gen, n):
  # Fit the model on the batches generated by datagen.flow_from_directory().
  model.fit_generator(
    training_gen,
    steps_per_epoch=int(
      (params["network"]["percent_per_epoch"] * n)
    / params["network"]["batch_size"]
    ),
    callbacks=[
      EarlyStopping(monitor="loss", min_delta=0, patience=0)
    ],
    epochs=params["network"]["epochs"]
  )

def rate_images(items, model):
  src = items["image"]
  if params["network"]["subtract_mean"]:
    src = items["image_deviation"]

  result = []
  print("There are {} example images.".format(items["count"]))
  progress = 0
  for i, img in enumerate(src):
    utils.prbar(i / items["count"])
    img = img.reshape((1,) + img.shape) # pretend it's a batch
    result.append(model.test_on_batch(img, img))

  print() # done with the progress bar
  return np.asarray(result)

def get_images(simple_gen):
  images = []
  classes = []
  print("There are {} example images.".format(items["count"]))

  for i in range(items["count"]):
    img, cls = next(simple_gen)
    images.append(img[0])
    classes.append(cls[0])

  return images, classes

def images_sorted_by_accuracy(items):
  return np.asarray([
    pair[0] for pair in
      sorted(
        list(
          zip(items["image"], items["rating"])
        ),
        key=lambda pair: pair[1]
      )
  ])

def save_images(images, directory, name_template, labels=None):
  for i in range(len(images)):
    l = str(labels[i]) if (not (labels is None)) else None
    img = toimage(images[i], cmin=0.0, cmax=1.0)
    convert_colorspace(
      img,
      params["network"]["training_colorspace"],
      params["network"]["initial_colorspace"]
    )
    fn = os.path.join(
      params["output"]["directory"],
      directory,
      name_template.format("{:03}".format(i))
    )
    img.save(fn)
    if l:
      subprocess.run([
        "mogrify",
          "-label",
          l,
          fn
      ])

def montage_images(directory, name_template, label=None):
  path = os.path.join(params["output"]["directory"], directory)
  targets = glob.glob(os.path.join(path, name_template.format("*")))
  targets.sort()
  output = os.path.join(path, name_template.format("montage"))
  if name_template.endswith("pdf"):
    subprocess.run([
      "gs",
        "-dBATCH",
        "-dNOPAUSE",
        "-q",
        "-sDEVICE=pdfwrite",
        "-sOutputFile={}".format(output),
    ] + targets
    )
  else:
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

def collect_montages(directory, label_dirnames=False):
  path = os.path.join(params["output"]["directory"], directory)
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


def get_features(images, model):
  encoder = get_encoding_model(model)
  return encoder.predict(np.asarray(images))

def save_training_examples(items, generator):
  print("  Saving training examples...")
  try:
    os.mkdir(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["transformed_dir"]
      ),
      mode=0o755
    )
  except FileExistsError:
    pass
  ex_input, ex_output = next(generator)
  ex_raw = items["image"][:len(ex_input)]
  save_images(
    ex_raw,
    params["filenames"]["transformed_dir"],
    params["filenames"]["raw_example"]
  )
  save_images(
    ex_input,
    params["filenames"]["transformed_dir"],
    params["filenames"]["input_example"]
  )
  save_images(
    ex_output,
    params["filenames"]["transformed_dir"],
    params["filenames"]["output_example"]
  )
  montage_images(
    params["filenames"]["transformed_dir"],
    params["filenames"]["raw_example"]
  )
  montage_images(
    params["filenames"]["transformed_dir"],
    params["filenames"]["input_example"]
  )
  montage_images(
    params["filenames"]["transformed_dir"],
    params["filenames"]["output_example"]
  )
  collect_montages(params["filenames"]["transformed_dir"])
  print("  ...done saving training examples...")

def analyze_duplicates(items):
  print("  Analyzing duplicates...")
  try:
    os.mkdir(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["duplicates_dir"]
      ),
      mode=0o755
    )
  except FileExistsError:
    pass

  items["duplicates"] = (
    items["distances"].shape[1] - np.count_nonzero(
      items["distances"],
      axis=1
    )
  )

  reps = []
  skip = set()
  for i, img in enumerate(items["image"]):
    utils.prbar(i / items["count"])
    if i not in skip and items["duplicates"][i] >= 2:
      reps.append(i)
      for j in range(items["distances"].shape[1]):
        if items["distances"][i][j] == 0:
          skip.add(j)
          print()

  representatives = items["image"][reps]
  duplications = items["duplicates"][reps]

  order = np.argsort(duplications)
  representatives = representatives[order]
  duplications = duplications[order]

  representatives = representatives[-params["output"]["large_pool_size"]:]
  duplications = duplications[-params["output"]["large_pool_size"]:]

  save_images(
    representatives,
    params["filenames"]["duplicates_dir"],
    params["filenames"]["duplicate_image"],
    labels=duplications
  )
  montage_images(
    params["filenames"]["duplicates_dir"],
    params["filenames"]["duplicate_image"]
  )

  plt.clf()
  n, bins, patches = plt.hist(items["duplicates"], 100)
  #plt.plot(bins)
  plt.xlabel("Number of Duplicates")
  plt.ylabel("Number of Images")
  plt.savefig(
    os.path.join(
      params["output"]["directory"],
      params["filenames"]["histogram"].format("duplicates")
    )
  )
  print("  ...done.")

def get_clusters(method, items, use, metric="euclidean"):
  results = {}
  print("Clustering images using {}...".format(method.__name__))
  print("  Using metric '{}'".format(metric))
  # Decide cluster input:
  print("  Using input '{}'".format(use))

  print("  Computing nearest-neighbor distances...")
  items["distances"] = pairwise.pairwise_distances(
    items[use],
    metric=metric
  )
  print("  ...done.")

  if "duplicates" in params["analysis"]["methods"]:
    print('-'*80)
    analyze_duplicates(items)

  # We want only nearest-neighbors for novel clustering (otherwise sorting
  # edges is too expensive).
  if method == NovelClustering:
    results["nearest_neighbors"] = np.argsort(
      items["distances"],
      axis=1
    )[:,1:params["clustering"]["neighborhood_size"]+1]
    results["neighbor_distances"] = np.zeros_like(
      results["nearest_neighbors"],
      dtype=float
    )
    for i, row in enumerate(results["nearest_neighbors"]):
      results["neighbor_distances"][i] = items["distances"][i][row]

  # If we're not using DBSCAN we don't need clustering_distance
  if method == DBSCAN:
    # Figure out what the distance value should be:
    print("  Computing DBSCAN cutoff distance...")

    # sort our distance array and take the first few as nearby points
    # offset by 1 excludes the zero distance to self
    # TODO: Why doesn't this work?!?
    results["ordered_distances"] = np.sort(
      items["distances"],
      axis=1
    )[:,1:params["clustering"]["dbscan_neighbors"]+1]
    results["outer_distances"] = results["ordered_distances"][
      :,
      params["clustering"]["dbscan_neighbors"]-1
    ]
    results["outer_distances"] = np.sort(results["outer_distances"])
    smp = results["outer_distances"][::items["count"]//10]
    print("   Distance sample:")
    print(smp)
    print("  ...done.")
    #closest, min_dist = pairwise.pairwise_distances_argmin_min(
    #  items[params["clustering"]["input"]],
    #  items[params["clustering"]["input"]],
    #  metric=metric
    #)
    clustering_distance = 0
    perc = params["clustering"]["dbscan_percentile"]
    while clustering_distance == 0 and perc < 100:
      clustering_distance = np.percentile(results["outer_distances"], perc)
      perc += 1

    if clustering_distance == 0:
      print(
        "Error determining clustering distance: all values were zero!",
        file=sys.stderr
      ) 
      exit(1)

    print(
      "  {}% {}th-neighbor distance is {}".format(
        perc-1,
        params["clustering"]["dbscan_neighbors"],
        clustering_distance
      )
    )

  if method == DBSCAN:
    model = DBSCAN(
      eps=clustering_distance,
      min_samples=params["clustering"]["dbscan_neighbors"],
      metric=metric,
      algorithm="auto"
    )
  elif method == AffinityPropagation:
    model = method(affinity=metric)
  elif method == AgglomerativeClustering:
    model = method(
      affinity=metric,
      linkage="average",
      n_clusters=2
    )
  # method "cluster" doesn't have a model

  print("  Clustering images...")
  result = None
  if method == NovelClustering:
    edges = [
      (
        fr,
        results["nearest_neighbors"][fr][tidx],
        results["neighbor_distances"][fr,tidx]
      )
        for fr in range(results["nearest_neighbors"].shape[0])
        for tidx in range(results["nearest_neighbors"].shape[1])
    ]
    results["cluster"] = method(
      items[params["clustering"]["input"]],
      edges=edges
    )
  else:
    fit = model.fit(items[params["clustering"]["input"]])
    results["cluster"] = fit.labels_

  if method == DBSCAN:
    results["core_mask"]= np.zeros_like(fit.labels_, dtype=int)
    results["core_mask"][fit.core_sample_indices_] = 1
    core_count = np.count_nonzero(results["core_mask"])
    print(
      "Core samples: {}/{} ({:.2f}%)".format(
        core_count,
        items["count"], 
        100 * core_count / items["count"]
      )
    )
  print("  ...done clustering.")

  results["cluster_ids"] = set(results["cluster"])
  unfiltered = len(results["cluster_ids"])

  results["cluster_sizes"] = {}
  for c in results["cluster"]:
    if c not in results["cluster_sizes"]:
      results["cluster_sizes"][c] = 1
    else:
      results["cluster_sizes"][c] += 1

  for i in range(len(results["cluster"])):
    if results["cluster_sizes"][results["cluster"][i]] == 1:
      results["cluster"][i] = -1

  results["cluster_ids"] = set(results["cluster"])
  if len(results["cluster_ids"]) != unfiltered:
    # Have to reassign cluster IDs:
    remap = {}
    new_id = 0
    for i in range(len(results["cluster"])):
      if results["cluster"][i] == -1:
        continue
      if results["cluster"][i] not in remap:
        remap[results["cluster"][i]] = new_id
        results["cluster"][i] = new_id
        new_id += 1
      else: 
        results["cluster"][i] = remap[results["cluster"][i]]

  results["cluster_sizes"] = {}
  for c in results["cluster"]:
    if c not in results["cluster_sizes"]:
      results["cluster_sizes"][c] = 1
    else:
      results["cluster_sizes"][c] += 1

  results["cluster_ids"] = set(results["cluster"])

  if -1 in results["cluster_ids"]:
    outlier_count = results["cluster_sizes"][-1]
    print(
      "  Found {} cluster(s) (with {:2.3f}% outliers)".format(
        len(results["cluster_ids"]) - 1,
        100 * (outlier_count / items["count"])
      )
    )
  else:
    print(
      "  Found {} cluster(s) (no outliers)".format(len(results["cluster_ids"]))
    )

  if method == DBSCAN:
    return (
      results["cluster"],
      results["cluster_ids"],
      results["cluster_sizes"],
      results["ordered_distances"],
      results["outer_distances"],
      results["core_mask"],
    )
  else:
    return (
      results["cluster"],
      results["cluster_ids"],
      results["cluster_sizes"],
    )
  print("  ...done clustering images.")

@utils.multilevel_default_params(DEFAULT_PARAMETERS)
def analyze_dataset(**params):
  """
  Analyzes a 
  """
  # Seed random number generator (hopefully improve image loading times via
  # disk cache?)
  print("Random seed is: {}".format(params["options"].seed))
  random.seed(params["options"].seed)
  # Backup old output directory and create a new one:
  print("Managing output backups...")
  bn = params["output"]["history_name"].format(
    params["output"]["keep_history"] - 1
  )
  if os.path.exists(bn):
    print(
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
      print("  ...found.")
      os.rename(bn, nbn)

  if os.path.exists(params["output"]["directory"]):
    bn = params["output"]["history_name"].format(0)
    shutil.make_archive(bn[:-4], 'zip', params["output"]["directory"])
    shutil.rmtree(params["output"]["directory"])

  try:
    os.mkdir(params["output"]["directory"], mode=0o755)
  except FileExistsError:
    pass
  print("  ...done.")

  # First load the CSV data:
  items = load_data()

  print('-'*80)
  print("Loading {} images...".format(items["count"]))
  #simple_gen = create_simple_generator()
  #images, classes = get_images(simple_gen)
  #images, classes = load_images_into_items(items)
  load_images_into_items(items)
  if "mean_image" in params["analysis"]["methods"]:
    print("  Saving mean image...")
    save_images([items["mean_image"]], ".", params["filenames"]["mean_image"])
  print("  ...done loading images.")
  print('-'*80)

  if params["options"].mode == "detect":
    params["options"].mode = utils.cached_value(
      lambda: "autoencoder",
      "mode",
      "str",
      debug=print
    )
  else:
    print("Selected mode '{}'".format(params["options"].mode))
    utils.store_cache(params["options"].mode, "mode", "str")

  if params["options"].mode == "dual":
    required_models = [
      "autoencoder-model",
      "predictor-model"
    ]
  else:
    required_models = [ params["options"].mode + "-model" ]

  print('-'*80)
  print("Acquiring {} model...".format(params["options"].mode))
  if params["options"].mode == "dual":
    def get_models():
      nonlocal items, params
      print("  Creating models...")
      inp, comp = setup_computation(items, mode="dual")
      ae_model = compile_model(inp, comp[0], mode="autoencoder")
      pr_model = compile_model(inp, comp[1], mode="predictor")
      print("  ...done creating models.")
      print("  Creating training generators...")
      ae_train_gen = create_training_generator(items, mode="autoencoder")
      pr_train_gen = create_training_generator(items, mode="predictor")
      print("  ...done creating training generators.")
      if "training_examples" in params["analysis"]["methods"]:
        save_training_examples(items, ae_train_gen)
      print("  Training models...")
      train_model(ae_model, ae_train_gen, items["count"])
      train_model(pr_model, pr_train_gen, items["count"])
      print("  ...done training models.")
      return (ae_model, pr_model)

    ae_model, pr_model = utils.cached_values(
      get_models,
      ("autoencoder-model", "predictor-model"),
      ("h5", "h5"),
      override=params["options"].model,
      debug=print
    )
  else:
    def get_model():
      nonlocal items, params
      print("  Creating model...")
      inp, comp = setup_computation(items, mode=params["options"].mode)
      model = compile_model(inp, comp, mode=params["options"].mode)
      print("  ...done creating model.")
      print("  Creating training generator...")
      train_gen = create_training_generator(items, mode=params["options"].mode)
      print("  ...done creating training generator.")
      if "training_examples" in params["analysis"]["methods"]:
        save_training_examples(items, train_gen)
      print("  Training model...")
      if params["options"].mode == "dual":
        train_model(ae_model, ae_train_gen, items["count"])
        train_model(pr_model, pr_train_gen, items["count"])
      else:
        train_model(model, train_gen, items["count"])
      print("  ...done training model.")
      return model

    model = utils.cached_value(
      get_model,
      params["options"].mode + "-model",
      typ="h5",
      override=params["options"].model,
      debug=print
    )

  print('-'*80)
  if params["options"].mode == "dual":
    print("Got models:")
    print(ae_model.summary())
    print("\n...and...\n")
    print(pr_model.summary())
  else:
    print("Got model:")
    print(model.summary())
  print('-'*80)

  if params["options"].mode == "dual":
    # DEBUG:
    test_autoencoder(items, ae_model, params["options"])
    test_predictor(items, pr_model, params["options"])
  elif options.mode == "autoencoder":
    test_autoencoder(items, model, params["options"])
  elif params["options"].mode == "predictor":
    test_predictor(items, model, params["options"])
  else:
    print(
      "Error: Unknown mode '{}'. No tests to run.".format(
        params["options"].mode
      ),
      file=sys.stderr
    )

def analyze_correlations(items, columns, against):
  p_threshold = 1 / (20 * len(columns))
  utils.reset_color()
  for col in columns:
    other = items[col]

    r, p = pearsonr(items[against], other)
    if p < p_threshold:
      print("  '{}': {}  (p={})".format(col, r, p))
      plt.clf()
      vtype = items["types"][col]
      if vtype == "boolean":
        # a histogram of proportions
        resolution = 50
        x = np.linspace(
          np.min(items[against]),
          np.max(items[against]),
          resolution
        )

        y = [] # array of binned proportions
        s = [] # array of bin counts
        for i in range(len(x)-1):
          if i == len(x)-2:
            # grab upper-extreme point:
            matches = (x[i] <= items[against]) & (items[against] <= x[i+1])
          else:
            matches = (x[i] <= items[against]) & (items[against] < x[i+1])
          total = sum(matches)
          s.append(total)
          if total != 0:
            y.append(sum(other[matches]) / total)
          else:
            y.append(-0.05) # nothing to count here; proportion is undefined
          other[matches]

        y = np.array(y)
        s = np.array(s)

        ns = s / np.max(s)

        #plt.bar(x[:-1], y, w, color=utils.pick_color())
        plt.scatter(x[:-1], y, c=utils.pick_color(), s=0.02 + 7.8*ns)
        plt.xlabel(against)
        plt.ylabel(col + " proportion")
      elif vtype in ("integer", "numeric"):
        # A histogram of means:
        resolution = 50
        x = np.linspace(
          np.min(items[against]),
          np.max(items[against]),
          resolution
        )

        y = [] # array of binned means
        s = [] # array of bin counts
        for i in range(len(x)-1):
          if i == len(x)-2:
            # grab upper-extreme point:
            matches = (x[i] <= items[against]) & (items[against] <= x[i+1])
          else:
            matches = (x[i] <= items[against]) & (items[against] < x[i+1])
          total = sum(matches)
          s.append(total)
          if total != 0:
            y.append(np.mean(other[matches]))
          else:
            y.append(-0.05) # nothing to count here; mean is undefined
          other[matches]

        y = np.array(y)
        s = np.array(s)

        ns = s / np.max(s)

        #plt.bar(x[:-1], y, w, color=utils.pick_color())
        plt.scatter(x[:-1], y, c=utils.pick_color(), s=0.02 + 7.8*ns)
        plt.xlabel(against)
        plt.ylabel(col + " proportion")
        # a density plot
        #resolution = 50
        #print("  ...binning {} vs. {} for plotting...".format(col, against))
        #matrix, xe, ye = np.histogram2d(
        #  items[against],
        #  other,
        #  bins=resolution,
        #  normed=True
        #)
        #matrix = np.transpose(matrix)
        #print("  ...done binning.")
        #print("  ...plotting binned items...")
        #plt.matshow(
        #  matrix,
        #  fignum=0,
        #  cmap=plt.get_cmap("YlGnBu"),
        #  origin="lower"
        #)
        #print("  ...done plotting.")
        plt.xlabel(against)
        plt.ylabel(col + " mean")
      else:
        # give up and do a scatterplot
        plt.scatter(items[against], other, s=0.25)
        plt.xlabel(against)
        plt.ylabel(col)

      plt.savefig(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["correlation_report"].format(against, col)
        )
      )
    else:
      print("    '{}': not significant (p={})".format(col, p))

  montage_images(
    ".",
    params["filenames"]["correlation_report"].format(against, "{}")
  )

def distribution_type(items, col):
  vtype = items["types"][col]
  if vtype == "boolean":
    return "binomial"
  elif vtype == "categorical":
    if len(items["values"][col]) == 2:
      return "binomial"
    else:
      raise RuntimeWarning(
"Warning: Using 'normalish' distribution for multi-category column '{}'."
        .format(col, vtype)
      )
      return "normalish"
  elif vtype in ("integer", "numeric"):
    # TODO: actually test this here?
    return "normalish"
  else:
    raise RuntimeWarning(
"Warning: Using 'normalish' distribution for column '{}' with type '{}'."
      .format(col, vtype)
    )
    return "normalish"

def relevant_statistic(items, col):
  vtype = items["types"][col]

  if vtype == "boolean":
    stat = simple_proportion
  elif vtype == "categorical":
    if len(items["values"][col]) == 2:
      stat = simple_proportion
    else:
      stat = np.average
  elif vtype in ("integer", "numeric"):
    stat = np.average
  else:
    stat = np.average
    raise RuntimeWarning(
"Warning: Using 'average' stat for column '{}' with type '{}'."
      .format(col, vtype)
    )
  return stat

def analyze_cluster_stats(items, which_stats):
  cstats = {
    c: {} for c in items["cluster_ids"]
  }

  for c in cstats:
    indices = items["cluster"] == c
    for col in which_stats:
      cstats[c][col] = items[col][indices]
    cstats[c]["size"] = len(cstats[c][which_stats[0]])

  plt.close()
  big_enough = [
    c
      for c in cstats
      if (
        cstats[c]["size"] >= params["clustering"]["significant_size"]
    and c != -1
      )
  ]
  big_enough = sorted(
    big_enough,
    key = lambda c: cstats[c]["size"]
  )

  # Synthesize per-cluster stats into per-column cinfo lists:
  cinfo = { "size": [] }
  nbig = len(big_enough)
  print(
    "  ...there are {} clusters above size {}...".format(
      nbig,
      params["clustering"]["significant_size"]
    )
  )

  # Compute desired confidence bounds:
  ntests = len(which_stats)
  total_tests = nbig * ntests

  # The Šidák correction:
  # TODO: Bonferroni correction? That requires a whole other algorithm though.
  shared_alpha = 1 - (
    1 - params["analysis"]["confidence_baseline"]
  )**(1 / total_tests)
  bootstrap_samples = int(2/shared_alpha)
  print(
    (
"  ...testing {} properties of {} clusters ({} total tests)...\n"
"  ...setting individual α={} for joint α={}...\n"
"  ...using {} samples for bootstrapping (2/α)..."
    ).format(
      ntests, nbig, total_tests,
      shared_alpha, params["analysis"]["confidence_baseline"],
      bootstrap_samples
    )
  )

  print("  ...extracting cluster stats...")
  for i, c in enumerate(big_enough):
    utils.prbar(i / len(big_enough))
    #for col in which_stats:
    for col in which_stats:
      if col not in cinfo:
        cinfo[col] = []

      x = cstats[c][col]

      stat = relevant_statistic(items, col)
      dist = distribution_type(items, col)

      cstats[c][col + "_stat"] = stat(x)
      cstats[c][col + "_ci"] = confidence_interval(
        x,
        dist,
        bound=shared_alpha,
        n_samples=bootstrap_samples
      )

      cinfo[col].append(
        (
          c,
          cstats[c][col + "_stat"],
          cstats[c][col + "_ci"][0],
          cstats[c][col + "_ci"][1]
        )
      )
    sz = cstats[c]["size"]
    cinfo["size"].append((c, sz, sz, sz))

  print()
  print("  ...done extracting cluster stats.")

  # Compute significant differences:
  small = { col: set() for col in which_stats }
  large = { col: set() for col in which_stats }
  diff = { col: set() for col in which_stats }
  overall_stats = {
    col: relevant_statistic(items, col)(items[col]) for col in which_stats
  }
  print("  ...bootstrapping overall means...")
  overall_cis = {}
  for i, col in enumerate(which_stats):
    utils.prbar(i / len(which_stats))
    overall_cis[col] = confidence_interval(
      items[col],
      distribution_type(items, col),
      bound=shared_alpha,
      n_samples=bootstrap_samples
    )
  print()
  print("  ...done bootstrapping overall means.")

  print("  ...computing cluster property significance...")
  for i, col in enumerate(which_stats):
    utils.prbar(i / len(which_stats))
    for c in big_enough:
      # std-divergence tests:
      if cstats[c][col + "_ci"][1] < overall_cis[col][0]:
        small[col].add(c)
      elif cstats[c][col + "_ci"][0] > overall_cis[col][1]:
        large[col].add(c)

      if items["types"][col] == "boolean":
        # Use Fisher's exact test
        in_and_true = len([x for x in cstats[c][col] if x])
        in_and_false = cstats[c]["size"] - in_and_true

        # TODO: Exclude cluster members?
        all_and_true = len([x for x in items[col] if x])
        all_and_false = items["count"] - all_and_true

        table = np.array(
          [
            [ in_and_true,  all_and_true  ],
            [ in_and_false, all_and_false ]
          ]
        )
        statname = "odds"
        stat, p = fisher_exact(table, alternative="two-sided")

      elif items["types"][col] == "categorical":
        # Use Pearson's chi-squared test
        values = sorted(list(items["values"][col].values()))
        # TODO: This normalization is *REALLY* sketchy and only vaguely related
        # to Laplace smoothing! Find another way of comparing things.
        nf = 1 / len(cstats[c][col])
        contingency = np.array(
          [
            [ nf + sum([v == val for v in cstats[c][col]]) for val in values ],
            [ nf + sum([v == val for v in items[col]]) for val in values ],
          ]
        )
        statname = "chi²"
        stat, p, dof, exp = chi2_contingency(contingency, correction=True)

      elif items["types"][col] in ("integer", "numeric"):
        # TODO: Something else for power-distributed variables?
        # Note: log-transformation has been applied to some stats above
        # Note: may require distribution-fitting followed by model
        # log-likelihood ratio analysis.
        stat, p = ttest_ind(
          cstats[c][col],
          items[col],
          equal_var=False, # Apply Welch's correction for non-equal variances
          nan_policy="omit" # shouldn't matter
        )
        statname = "t"

      else:
        # Don't know how to analyze this statistically
        raise RuntimeWarning(
"Warning: Column '{}' with type '{}' can't be compared against population."
          .format(
            col,
            items["types"][col]
          )
        )

      if p < shared_alpha:
        diff[col].add(c)

  print()
  print("  ...done computing cluster property significance.")

  print("  ...plotting & summarizing cluster stats...")
  for col in cinfo:
    if not cinfo[col]:
      continue

    with open(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["cluster_stats"].format("outliers")[:-4] + ".txt"
      ),
      'a'
    ) as fout:
      if col in which_stats and any((small[col], large[col], diff[col])):
        print("    ...{} is significant...".format(col))
        fout.write("Outliers for '{}':\n".format(col))
        fout.write(
          "  small:" + ", ".join(str(x) for x in sorted(list(small[col]))) +"\n"
        )
        fout.write(
          "  large:" + ", ".join(str(x) for x in sorted(list(large[col]))) +"\n"
        )
        fout.write(
          "  diff:" + ", ".join(str(x) for x in sorted(list(diff[col]))) + "\n"
        )

    cinfo[col] = np.asarray(cinfo[col])
    # x values are just integers:
    x = np.arange(nbig)
    cids = cinfo[col][:,0]
    stats = cinfo[col][:,1]
    bot = cinfo[col][:,2]
    top = cinfo[col][:,3]
    if col in which_stats:
      difference = np.asarray([int(int(cid) in diff[col]) for cid in cids])
      size = (
        -np.asarray([int(int(cid) in small[col]) for cid in cids])
      + np.asarray([int(int(cid) in large[col]) for cid in cids])
      )
    else:
      size = [0] * len(cids)
      difference = [0] * len(cids)

    colors = np.asarray(
      [
        ((0, 0, 0), (0, 0.1, 0.6))[difference[i]]
        #((0, 0, 0), (1, 0, 0), (0, 0, 1))[size[i]]
          for i in x
      ]
    )

    plt.clf()

    # plot lines out to the standard deviations:
    plt.vlines(
      x,
      bot,
      top,
      lw=0.6,
      colors=colors
    )
    if col in overall_stats:
      plt.axhline(overall_cis[col][0], lw=0.6, c="k")
      plt.axhline(overall_cis[col][1], lw=0.6, c="k")
    # plot the stats:
    plt.scatter(x, stats, s=0.9, c=colors)

    plt.title("{} ({} clusters)".format(col, nbig))
    plt.xlabel("cluster")
    if col in items["types"]:
      plt.ylabel(relevant_statistic(items, col).__name__)
    else:
      plt.ylabel(col)
    plt.savefig(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["cluster_stats"].format(col)
      )
    )
  print("  ...done plotting & summarizing.")

  montage_images(".", params["filenames"]["cluster_stats"])

def reconstruct_image(items, img, model):
  img = img.reshape((1,) + img.shape) # pretend it's a batch
  pred = model.predict(img)[0]
  if params["network"]["subtract_mean"]:
    pred += items["mean_image"]
  return pred

def test_autoencoder(items, model, params):
  items["rating"] = utils.cached_value(
    lambda: rate_images(items, model),
    "ratings",
    override=params["options"].rank,
    debug=print
  )
  print('-'*80)
  items["norm_rating"] = items["rating"] / np.max(items["rating"])
  items["values"]["rating"] = "numeric"
  items["types"]["rating"] = "numeric"
  items["values"]["norm_rating"] = "numeric"
  items["types"]["norm_rating"] = "numeric"

  # Save the best images and their reconstructions:
  if "reconstructions" in params["analysis"]["methods"]:
    print("Saving example images...")
    try:
      os.mkdir(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["examples_dir"]
        ),
        mode=0o755
      )
    except FileExistsError:
      pass
    sorted_images = images_sorted_by_accuracy(items)

    best = sorted_images[:params["output"]["example_pool_size"]]
    worst = sorted_images[-params["output"]["example_pool_size"]:]
    rnd = items["image"][:]
    random.shuffle(rnd)
    rnd = rnd[:params["output"]["example_pool_size"]]

    for iset, fnt in zip(
      [best, worst, rnd],
      [
        params["filenames"]["best_image"],
        params["filenames"]["worst_image"],
        params["filenames"]["sampled_image"]
      ]
    ):
      save_images(iset, params["filename"]["examples_dir"], fnt)
      rec_images = []
      for img in iset:
        rec_images.append(reconstruct_image(items, img, model))
      save_images(rec_images, params["filename"]["examples_dir"], "rec-" + fnt)
      montage_images(params["filename"]["examples_dir"], fnt)
      montage_images(params["filename"]["examples_dir"], "rec-" + fnt)
    collect_montages(params["filename"]["examples_dir"])
    print("  ...done.")

  print('-'*80)
  src = items["image"]
  if params["network"]["subtract_mean"]:
    src = items["image_deviation"]
  items["features"] = utils.cached_value(
    lambda: get_features(src, model),
    "features",
    override=params["options"].features,
    debug=print
  )

  print('-'*80)
  tsne = TSNE(n_components=2, random_state=0)
  items["projected"] = utils.cached_value(
    lambda: tsne.fit_transform(items["features"]),
    "projected",
    override=params["options"].project,
    debug=print
  )


  print('-'*80)
  if params["clustering"]["method"] == DBSCAN:
    (
      items["cluster"],
      items["cluster_ids"],
      items["cluster_sizes"],
      items["ordered_distances"],
      items["outer_distances"],
      items["core_mask"],
    ) = utils.cached_values(
      lambda:
        get_clusters(
          params["clustering"]["method"],
          items,
          params["clustering"]["input"],
          params["clustering"]["metric"]
        ),
      (
        "clusters",
        "cluster_ids",
        "cluster_sizes",
        "ordered_distances",
        "outer_distances",
        "core_mask",
      ),
      ("pkl", "pkl", "pkl", "pkl", "pkl", "pkl"),
      override=params["options"].cluster,
      debug=print
    )
  else:
    (
      items["cluster"],
      items["cluster_ids"],
      items["cluster_sizes"],
    ) = utils.cached_values(
      lambda: get_clusters(
        params["clustering"]["method"],
        items,
        params["clustering"]["input"]
      ),
      ("clusters", "cluster_ids", "cluster_sizes"),
      ("pkl", "pkl", "pkl"),
      override=params["options"].cluster,
      debug=print
    )

  # Plot a histogram of error values for all images:
  if "reconstruction_error" in params["analysis"]["methods"]:
    print('-'*80)
    print("Plotting reconstruction error histogram...")
    print("  Error limits:", np.min(items["rating"]), np.max(items["rating"]))
    plt.clf()
    n, bins, patches = plt.hist(items["rating"], 100)
    #plt.plot(bins)
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Number of Images")
    #plt.axis([0, 1.1*max(items["rating"]), 0, 1.2 * max(n)])
    #plt.show()
    plt.savefig(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["histogram"].format("error")
      )
    )
    plt.clf()
    print("  ...done.")

  if "reconstruction_correlations" in params["analysis"]["methods"]:
    print('-'*80)
    print("Computing reconstruction correlations...")
    analyze_correlations(
      items,
      params["analysis"]["correlate_with_error"],
      "norm_rating"
    )
    print("  ...done.")

  print('-'*80)
  print("Computing typicality...")
  items["typicality"] = utils.cached_value(
    lambda: typicality(
      items["features"],
      quiet=False,
      metric=params["clustering"]["metric"],
      significant_fraction=params["clustering"]["typicality_fraction"],
    ),
    "typicality",
    "pkl",
    override=params["options"].typicality,
    debug=print
  )
  items["values"]["typicality"] = "numeric"
  items["types"]["typicality"] = "numeric"
  print("  ...done.")
  if "typicality_correlations" in params["analysis"]["methods"]:
    print("Analyzing typicality...")
    analyze_correlations(
      items,
      params["analysis"]["correlate_with_error"],
      "typicality"
    )
    print("  ...done.")

  # Plot the t-SNE results:
  if "tSNE" in params["analysis"]["methods"]:
    print('-'*80)
    print("Plotting t-SNE results...")
    cycle = palettable.tableau.Tableau_20.mpl_colors
    colors = []
    alt_colors = []
    first_target = params["network"]["predict_target"][0]

    # subsample indices:
    if items["count"] > tsne_subsample:
      ss = np.random.choice(
        items["count"],
        params["analysis"]["tsne_subsample"],
        replace=False
      )
    else:
      ss = np.arange(items["count"])

    # Determine colors:
    vtype = items["types"][first_target]
    tv = items[first_target]
    sv = tv[ss]
    if vtype in ["numeric", "integer"]:
      norm = (sv - np.min(tv)) / (np.max(tv) - np.min(tv))
      cmap = plt.get_cmap("plasma")
      #cmap = plt.get_cmap("viridis")
      for v in norm:
        colors.append(cmap(v))
    elif vtype == "boolean":
      for v in sv:
        colors.append(cycle[v % len(cycle)])
    else: # including categorical items
      mapping = {}
      max_so_far = 0
      for v in sv:
        if type(v) == np.ndarray:
          v = tuple(v)
        if v in mapping:
          c = mapping[v]
        else:
          c = max_so_far
          max_so_far += 1
          mapping[v] = c
        colors.append(cycle[c % len(cycle)])

    sizes = 0.25
    if params["clustering"]["method"] == DBSCAN:
      sizes = items["core_mask"][ss]*0.75 + 0.25

    for cl in items["cluster"][ss]:
      if cl == -1:
        alt_colors.append((0.0, 0.0, 0.0)) # black
      else:
        alt_colors.append(cycle[cl % len(cycle)])

    if "typicality" in items:
      typ_colors = []
      cmap = plt.get_cmap("plasma")
      for t in items["typicality"][ss]:
        typ_colors.append(cmap(t))

    axes = [(0, 1)]
    if items["projected"].shape[1] == 3:
      axes = [(0, 1), (0, 2), (1, 2)]

    for i, dims in enumerate(axes):
      utils.prbar(i / len(dims))
      # Plot using true colors:
      x, y = dims

      plt.clf()
      ax = plt.scatter(
        items["projected"][ss,x],
        items["projected"][ss,y],
        s=0.25,
        c=colors
      )
      plt.xlabel("t-SNE {}".format("xyz"[x]))
      plt.ylabel("t-SNE {}".format("xyz"[y]))
      plt.savefig(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["tsne"].format("true", x, y)
        )
      )

      # Plot using guessed colors:
      plt.clf()
      ax = plt.scatter(
        items["projected"][ss,x],
        items["projected"][ss,y],
        s=sizes,
        c=alt_colors
      )
      plt.xlabel("t-SNE {}".format("xyz"[x]))
      plt.ylabel("t-SNE {}".format("xyz"[y]))
      plt.savefig(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["tsne"].format("learned", x, y)
        )
      )

      # Plot typicality:
      if "typicality" in items:
        plt.clf()
        ax = plt.scatter(
          items["projected"][ss,x],
          items["projected"][ss,y],
          s=0.25,
          c=typ_colors
        )
        plt.xlabel("t-SNE {}".format("xyz"[x]))
        plt.ylabel("t-SNE {}".format("xyz"[y]))
        plt.savefig(
          os.path.join(
            params["output"]["directory"],
            params["filenames"]["tsne"].format("typ", x, y)
          )
        )

    print()
    # TODO: Less hacky here
    montage_images(".", params["filenames"]["tsne"].format("*", "*", "{}"))
    print("  ...done.")

  # Plot a histogram of pairwise distance values (if we computed them):
  if (
    "distance_histograms" in params["analysis"]["methods"]
and params["clustering"]["method"] == DBSCAN
  ):
    print('-'*80)
    print(
      "Plotting distance histograms...".format(
        params["clustering"]["dbscan_neighbors"]
      )
    )

    for col in range(items["ordered_distances"].shape[1]):
      #n, bins, patches = plt.hist(
      #  items["ordered_distances"][:,col],
      #  1000,
      #  cumulative=True
      #)
      n, bins, patches = plt.hist(items["ordered_distances"][:,col], 1000)
      #plt.plot(bins)
      plt.xlabel("Distance to {} Neighbor".format(ordinal(col+1)))
      plt.ylabel("Number of Images")
      #plt.axis([0, 1.1*max(items["outer_distances"]), 0, 1.2 * max(n)])
      plt.savefig(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["histogram"].format("distance-{}".format(col))
        )
      )
      plt.clf()
    montage_images(".", params["filenames"]["histogram"].format("distance-{}"))
    print("  ...done.")

  if "distances" in params["analysis"]["methods"]:
    print('-'*80)
    print("Plotting distances...")
    plt.plot(items["outer_distances"])
    plt.xlabel("Index")
    plt.ylabel(
      "Distance to {} Neighbor".format(
        ordinal(params["clustering"]["dbscan_neighbors"])
      )
    )
    plt.savefig(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["distances"].format(
          ordinal(params["clustering"]["dbscan_neighbors"])
        )
      )
    )
    plt.clf()

    ods = items["ordered_distances"]
    distance_ratios = []
    skipped = {}
    for i in range(ods.shape[0]):
      for j in range(ods.shape[1] - 1):
        if ods[i, j] > 0:
          distance_ratios.append(ods[i, j+1] / ods[i, j])
        else:
          if j in skipped:
            skipped[j] += 1
          else:
            skipped[j] = 1

    print("  Number of nth-neighbor clones:")
    for k in sorted(list(skipped.keys())):
      print("    {}: {}".format(k, skipped[k]))

    if distance_ratios:
      n, bins, patches = plt.hist(distance_ratios, 1000)
      plt.xlabel("Distance ratio between 1st and 2nd Neighbors")
      plt.ylabel("Number of Images")
      plt.savefig(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["histogram"].format("distance-ratio")
        )
      )
      plt.clf()
    else:
      raise RuntimeWarning(
        "Warning: no distance ratio information available (too many clones)."
      )
    print("  ...done.")

  if "cluster_sizes" in params["analysis"]["methods"]:
    print('-'*80)
    # Plot cluster sizes
    print("Plotting cluster sizes...")
    just_counts = sorted(list(items["cluster_sizes"].values()))
    plt.clf()
    plt.scatter(range(len(just_counts)), just_counts, s=2.5, c="k")
    plt.xlabel("cluster")
    plt.ylabel("cluster size")
    plt.savefig(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["cluster_sizes"]
      )
    )
    plt.clf()
    with open(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["cluster_sizes"][:-4] + ".txt"
      ),
      'a'
    ) as fout:
      both_by_size = sorted(
        list(items["cluster_sizes"].items()),
        key=lambda kv: kv[1]
      )
      fout.write(", ".join(str(k) for (k, v) in both_by_size) + "\n")
      fout.write(", ".join(str(v) for (k, v) in both_by_size) + "\n")
    print("  ...done.")

  if "cluster_statistics" in params["analysis"]["methods"]:
    print('-'*80)
    # Summarize statistics per-cluster:
    print("Summarizing clustered statistics...")
    analyze_cluster_stats(
      items,
      params["analysis"]["analyze_per_cluster"],
      SECONDARY_PER_CLUSTER
    )
    print("  ...done.")

  if "cluster_samples" in params["analysis"]["methods"]:
    print('-'*80)
    # Show some of the clustering results (TODO: better):
    print("Sampling clustered images...")
    try:
      os.mkdir(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["clusters_dir"]
        ),
        mode=0o755
      )
    except FileExistsError:
      pass
    try:
      os.mkdir(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["clusters_rec_dir"]
        ),
        mode=0o755
      )
    except FileExistsError:
      pass

    # TODO: Get more representative images?
    viz = sorted([
      k
        for (k, v) in items["cluster_sizes"].items()
        if v > params["clustering"]["viz_size"]
    ])[-params["output"]["max_cluster_samples"]-1:]
    nvc = len(viz)
    shuf = list(zip(items["image"], items["cluster"]))
    random.shuffle(shuf)

    for i, c in enumerate(viz):
      utils.prbar(i / nvc)
      cluster_images = []
      rec_images = []
      for i, (img, cluster) in enumerate(shuf):
        if cluster == c:
          rec = reconstruct_image(items, img, model)
          cluster_images.append(img)
          rec_images.append(rec)
          if len(cluster_images) >= params["output"]["samples_per_cluster"]:
            break
      if c == -1:
        thisdir = os.path.join(
          params["filenames"]["clusters_dir"],
          "outliers_({})".format(items["cluster_sizes"][c] + 1)
        )
        recdir = os.path.join(
          params["filenames"]["clusters_rec_dir"],
          "outliers_({})".format(items["cluster_sizes"][c] + 1)
        )
      else:
        thisdir = os.path.join(
          params["filenames"]["clusters_dir"],
          "cluster-{}_({})".format(c, items["cluster_sizes"][c] + 1)
        )
        recdir = os.path.join(
          params["filenames"]["clusters_rec_dir"],
          "cluster-{}_({})".format(c, items["cluster_sizes"][c] + 1)
        )
      try:
        os.mkdir(
          os.path.join(
            params["output"]["directory"],
            thisdir
          ),
          mode=0o755
        )
      except FileExistsError:
        pass
      try:
        os.mkdir(
          os.path.join(
            params["output"]["directory"],
            recdir
          ),
          mode=0o755
        )
      except FileExistsError:
        pass
      save_images(cluster_images, thisdir, params["filenames"]["cluster_rep"])
      save_images(rec_images, recdir, params["filenames"]["cluster_rep"])
      montage_images(
        thisdir,
        params["filenames"]["cluster_rep"],
        label=items["cluster_sizes"][c]
      )
      montage_images(
        recdir,
        params["filenames"]["cluster_rep"],
        label=items["cluster_sizes"][c]
      )

    print() # done with the progress bar
    print("  ...creating combined cluster sample image...")
    collect_montages(params["filenames"]["clusters_dir"], label_dirnames=True)
    collect_montages(
      params["filenames"]["clusters_rec_dir"],
      label_dirnames=True
    )
    print("  ...done.")

def test_predictor(items, model, options):
  if "prediction_accuracy" in params["analysis"]["methods"]:
    print('-'*80)
    print("Analyzing prediction accuracy...")
    print("  ...there are {} samples...".format(items["count"]))
    true = np.stack([items[t] for t in params["network"]["predict_target"]])

    src = items["image"]
    if params["network"]["subtract_mean"]:
      src = items["image_deviation"]

    rpred = model.predict(src)
    predicted = np.asarray(rpred.reshape(true.shape), dtype=float)

    for i, t in enumerate(params["analysis"]["predict_analysis"]):
      target = params["network"]["predict_target"][i]
      x = true[i,:]
      y = predicted[i,:]

      if t == "confusion":
        plt.clf()
        x = x > 0.5
        y = y > 0.5
        cm = confusion_matrix(x, y)
        plot_confusion_matrix(
          cm,
          list(set(x)),
          normalize=False,
          title=target.title()
        )
        plt.savefig(
          os.path.join(
            params["output"]["directory"],
            params["filenames"]["analysis"].format(target)
          )
        )
      elif t == "scatter":
        plt.clf()
        plt.scatter(x, y, s=0.25)
        fit = np.polyfit(x, y, deg=1)
        plt.plot(x, fit[0]*x + fit[1], color="red", linewidth=0.1)
        plt.xlabel("True {}".format(target.title()))
        plt.ylabel("Predicted {}".format(target.title()))
        plt.savefig(
          os.path.join(
            params["output"]["directory"],
            params["filenames"]["analysis"].format(target)
          )
        )

    print(" ...done.")


def plot_confusion_matrix(
  cm,
  classes,
  normalize=False,
  title='Confusion matrix',
  cmap=plt.cm.Blues
):
  """
  Confusion matrix code from:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run a CNN for image analysis.")
  parser.add_argument(
    "-M",
    "--mode",
    choices=["detect", "autoencoder", "predictor", "dual"],
    default="detect",
    help="""\
What kind of model to build & train. Options are:
(0) detect - detects previously used mode (the default)
(1) autoencoder - learns essential features without supervision
(2) predictor - learns to predict output variable(s)
"""
  )
  parser.add_argument(
    "-m",
    "--model",
    action="store_true",
    help="Recompute the model even if a cached model is found."
  )
  parser.add_argument(
    "-r",
    "--rank",
    action="store_true",
    help="Recompute rankings even if cached rankings are found."
  )
  parser.add_argument(
    "-f",
    "--features",
    action="store_true",
    help="Recompute features even if cached features are found."
  )
  parser.add_argument(
    "-j",
    "--project",
    action="store_true",
    help="Recompute t-SNE projection even if a cached projection is found."
  )
  parser.add_argument(
    "-c",
    "--cluster",
    action="store_true",
    help="Recompute clusters even if a cached clustering is found."
  )
  parser.add_argument(
    "-t",
    "--typicality",
    action="store_true",
    help="Recompute typicality even if cached typicality is found."
  )
  parser.add_argument(
    "-F",
    "--fresh",
    action="store_true",
    help="Recompute everything. Equivalent to '-mrfjc'."
  )
  parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=23,
    help="Set the random seed."
  )
  options = parser.parse_args()
  if options.fresh:
    options.model = True
    options.rank = True
    options.features = True
    options.project = True
    options.cluster = True
    options.typicality = True
    # Explicitly disable all caching
    utils.toggle_caching(False)

  print(dir(options))
  exit(1)
  analyze_dataset(options=options, quiet=False)
  #utils.run_strict(analyze_dataset, **options)

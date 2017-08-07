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
import warnings
import random
import shlex
import math
import csv
import re

import utils
import impr

import numpy as np

def import_libraries():
  """
  Imports the data processing libraries. Some of these *cough* keras *cough*
  take a bit of time to import, so we want to let the user know what's going
  on, but that requires printing, which in turn requires option parsing (so we
  know whether or not to print stuff) so... these end up in a function.
  """
  global t, pearsonr, fisher_exact, chi2_contingency, ttest_ind, \
    proportion_confint, ci, InstabilityWarning, ImageDataGenerator, \
    to_categorical, Model, EarlyStopping, Input, Dense, Flatten, Reshape, \
    Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, l1, plt, TSNE, \
    DBSCAN, AffinityPropagation, AgglomerativeClustering, pairwise, \
    confusion_matrix, imread, imsave, img_as_float, convert_colorspace, \
    palettable, neighbors_from_distances, get_neighbor_edges, \
    condensed_multiscale, cluster_assignments, typicality, isolation, \
    derive_isolation, reassign_cluster_ids, find_exemplars, \
    find_representatives

  from scipy.stats import t
  from scipy.stats import pearsonr
  from scipy.stats import fisher_exact
  from scipy.stats import chi2_contingency
  from scipy.stats import ttest_ind

  from statsmodels.stats.proportion import proportion_confint

  from scikits.bootstrap import ci
  from scikits.bootstrap import InstabilityWarning

  debug("Importing keras...")
  import keras
  from keras.preprocessing.image import ImageDataGenerator
  from keras.utils.np_utils import to_categorical
  from keras.models import Model
  from keras.callbacks import EarlyStopping
  from keras.layers import Input
  from keras.layers import Dense, Flatten, Reshape
  from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
  from keras.layers.normalization import BatchNormalization
  from keras.regularizers import l1
  debug("...done.")

  debug("Importing matplotlib...")
  import matplotlib.pyplot as plt
  debug("...done.")

  debug("Importing scikit-learn...")
  from sklearn.manifold import TSNE
  from sklearn.cluster import DBSCAN
  from sklearn.cluster import AffinityPropagation
  from sklearn.cluster import AgglomerativeClustering
  from sklearn.metrics import pairwise
  from sklearn.metrics import confusion_matrix
  debug("...done.")

  from skimage import img_as_float
  from skimage.io import imread
  from skimage.io import imsave
  from skimage.color import convert_colorspace

  import palettable

  from multiscale import neighbors_from_distances
  from multiscale import get_neighbor_edges
  from multiscale import condensed_multiscale
  from multiscale import cluster_assignments
  from multiscale import typicality
  from multiscale import isolation
  from multiscale import derive_isolation
  from multiscale import reassign_cluster_ids
  from multiscale import find_exemplars
  from multiscale import find_representatives

#------------------------------#
# Shims for imported functions #
#------------------------------#

def NovelClustering(points, distances=None, edges=None):
  """
  Wrapper around the stuff we imported from the multiscale library to make it
  fit as a clustering method here.
  """
  all_clusters = condensed_multiscale(
    points,
    distances=distances,
    edges=edges,
    quiet=False,
    use_cached_neighbors=False
  )
  #return cluster_assignments(points, all_clusters)
  return all_clusters # already have ids reassigned

def build_clustering(assignments):
  """
  Turns a cluster-id-assignment list into a list-of-cluster-dictionaries.
  """
  clusters = {}
  for p, a in enumerate(assignments):
    if a in clusters:
      clusters[a]["vertices"].add(p)
      clusters[a]["size"] += 1
    else:
      clusters[a] = {
        "vertices": { p },
        "size": 0 # number of "edges" = # of points - 1
      }
  return reassign_cluster_ids(clusters, order_by="size")

def simple_proportion(data):
  """
  Function for getting the proportion of a binary data column that's True. Can
  be used with bootstrapping to get confidence intervals for the true
  proportion.
  """
  return np.sum(data) / len(data)

def confidence_interval(
  data,
  distribution="normalish",
  bound=0.05,
  n_samples=10000,
  bootstrap_cutoff=15
):
  """
  Gets a confidence interval for the given data assuming the given underlying
  distribution (which can be either "normalish" or "binomial"). The result is
  in terms of the appropriate statistic (proportion for binomial data, or mean
  for normal-ish data). If the sample size is below the bootstrap_cutoff,
  bootstrapping will be used instead of a statistical assmption, although this
  has its own problems. That's the only case where n_samples comes into play
  (it sets the number of samples for bootstrapping). The computed confidence
  interval leaves out the given percentage of the estimated probability space
  on both sides: so the default bound of 0.05 represents ad 2.5%--97.5%
  confidence interval, with a total of 95% confidence that the true value is
  within the given bounds.
  """
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
  """
  A thin wrapper around ci from the bootstrap package, with a check to make
  sure that the data aren't singular.
  """
  method = "bca"
  if all(d == data[0] for d in data):
    # zero-variance case, fall back on a simple percentile interval
    method = "pi"

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
  "options": {
    "mode": "autoencoder",
    "model": False,
    "rank": False,
    "features": False,
    "project": False,
    "cluster": False,
    "typicality": False,
    "isolation": False,
    "fresh": False,
    "quiet": True,
    "seed": 23,
  },

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
    #"filter_on": [],
    "filter_on": [ "private" ],
    #"filter_on": [ "!private" ],
    #"filter_on": [ "!private", "!no-friends" ],
    "subset_size": 50000,
    #"subset_size": 10000,
  },

  "network": {
    # training parameters
    "batch_size": 32,
    "percent_per_epoch": 1.0, # how much of the data to feed per epoch
    #"epochs": 200,
    "epochs": 50,
    #"epochs": 10,
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
    #"metric": "cosine", # metric for distance measurement

    # typicality parameters:
    "typicality_fraction": 0.008, # percent of points to use for typicality

    #"method": AffinityPropagation,
    #"method": DBSCAN,
    #"method": AgglomerativeClustering,
    "method": NovelClustering,
    "input": "features",
    #"input": "projected",

    "neighborhood_size": 10,

    "dbscan_neighbors": 3,
    "dbscan_percentile": 80,

    "significant_size": 20,
    "viz_size": 10,
  },

  "projection": {
    "metric": "euclidean",
    #"metric": "cosine",

    #"target_dimensionality": 2,
    "target_dimensionality": 3,
  },

  # statistical analysis parameters:
  "analysis": {
    "double_check_ratings": False,
    "spot_check_typicality": False,

    "confidence_baseline": 0.05, # base desired confidence across all tests

    "methods": [
      "mean_image",
      #"training_examples",
      "reconstructions",
      "lineups",
      "reconstruction_error",
      "reconstruction_correlations",
      "typicality_correlations",
      #"isolation_correlations",
      "tSNE",
      #"distance_histograms",
      "nearest",
      "representatives",
      "representative_statistics",
      "exemplars",
      #"distances",
      "duplicates",
      "cluster_sizes",
      "cluster_statistics",
      "cluster_samples",
      "prediction_accuracy",
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
      "genres()",
    ] + [
      "genres({})".format(g) for g in ALL_GENRES
    ],

    "analyze_per_representative": [
      "norm_rating",
      "typicality",
      "country-code-US",
      "competence-Beginner",
      "competence-Expert",
      "competence-Intermediate",
      "log-friends",
      "log-following",
      "log-followers",
      "log-posts",
      "log-yeahs",
      "genres()",
    ] + [
      "genres({})".format(g) for g in ALL_GENRES
    ],

    "analyze_per_cluster": [
      "norm_rating",
      "typicality",
      "country-code-US",
      "competence-Beginner",
      "competence-Expert",
      "competence-Intermediate",
      "log-friends",
      "log-following",
      "log-followers",
      "log-posts",
      "log-yeahs",
      "genres()",
    ] + [
      "genres({})".format(g) for g in ALL_GENRES
    ],

    "examine_exemplars": [
      "genres(Sports)",
      "genres(Music)",
      "country-code-US",
      "competence"
    ],

    "predict_analysis": [ "confusion" ],

    "tsne_subsample": 2000,
    #"tsne_subsample": 10000,

    "image_lineup_steps": 25,
    "image_lineup_samples": 6,
    #"image_lineup_samples": 16,
    "show_lineup": [
      "norm_rating",
      "typicality",
      "isolation",
      "log-posts",
      "feature-0",
      "feature-1",
      "feature-2",
      "feature-3",
    ]
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
    "image_lineup": "image-lineup-{}.png",

    "correlation_report": "correlation-{}-{}.pdf",
    "histogram": "histogram-{}.pdf",
    "cluster_sizes": "cluster-sizes.pdf",
    "cluster_stats": "{}-cluster-stats-{}.pdf",
    "distances": "distances-{}.pdf",
    "tsne": "tsne-{}-{}x{}.pdf",
    "analysis": "analysis-{}.pdf",
    "cluster_rep": "rep-{}.png",
    "clusters_summary": "cluster-summary-{}.png",
    "nearest": "nearest-{}.png",
    "exemplar": "exemplar-{}-{}.png",
    "representative": "representative-{}-{}.png",

    "transformed_dir": "transformed",
    "examples_dir": "examples",
    "duplicates_dir": "duplicates",
    "nearest_dir": "nearest",
    "exemplars_dir": "exemplars",
    "representatives_dir": "representatives",
    "clusters_dir": "clusters",
    "clusters_rec_dir": "rec_clusters",
  },
}

def load_data(params):
  """
  Loads the data from the designated input file(s).
  """
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
  debug("Reading CSV file...")
  with open(params["input"]["csv_file"], 'r', newline='') as fin:
    reader = csv.reader(fin, dialect="excel")
    legend = next(reader)
    for i, key in enumerate(legend):
      if key in params["data_processing"]["field_types"]:
        types[key] = params["data_processing"]["field_types"][key]
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
        debug(lst, file=sys.stderr)
        debug("Ignoring unparsable line(s).", file=sys.stderr)

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

  debug("  ...found {} records..".format(len(full_items)))
  debug("  ...done.")

  debug("Expanding multi-value fields...")
  for col in legend:
    if types[col] == "multi":
      for v in values[col]:
        values["{}({})".format(col, v)] = "boolean"
        types["{}({})".format(col, v)] = "boolean"

  lfi = len(full_items)
  for i, (ikey, record) in enumerate(full_items.items()):
    utils.prbar(i / lfi, debug=debug)
    for col in legend:
      if types[col] == "multi":
        hits = record[col].split(
          params["data_processing"]["multi_field_separator"]
        )
        for v in values[col]:
          nfn = "{}({})".format(col, v)
          record[nfn] = v in hits

  debug("\n  ...done.")

  debug("Converting & filtering data...")
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
  debug(
    "  Filtered {} items down to {} accepted items...".format(precount, count)
  )

  if count > params["data_processing"]["subset_size"]:
    debug(
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
  debug("  ...done.")

  oldcols = list(long_items["types"].keys())
  for col in oldcols:
    if long_items["types"][col] == "categorical":
      cname = col + "-categorical"
      long_items[cname] = to_categorical(long_items[col])
      long_items["values"][cname] = "one-hot"
      long_items["types"][cname] = "one-hot"

  return long_items

def add_norm_column(items, col):
  """
  Adds a normalized version of the given column to the items. The name of the
  new column is the name of the original column with "-norm" appended.
  """
  nname = col + "-norm"
  items["values"][nname] = "numeric"
  items["types"][nname] = "numeric"
  col_max = np.max(items[col])
  items[nname] = items[col] / col_max

def add_log_column(items, col):
  """
  Adds a log-transformed version of the given column to the items. The name of
  the new column is the name of the original column with "log-" prepended.
  """
  nname = "log-" + col
  items["values"][nname] = "numeric"
  items["types"][nname] = "numeric"
  #items[nname] = np.log(items[col] + np.min(items[col][items[col]!=0])/2)
  items[nname] = np.log(items[col] + 1)

def add_binary_column(items, col, val, name):
  """
  Adds a version of the given column that just has True where values are equal
  to the given 'val' and False elsewhere. The new column is given the requested
  name (careful that this doesn't collide and end up deleting a column).
  """
  items["values"][name] = "boolean"
  items["types"][name] = "boolean"
  items[name] = items[col] == val

def setup_computation(items, params, mode="autoencoder"):
  """
  Given the input data, sets up a neural network using Keras, and returns an
  (input, output) pair of computation graph nodes. If the mode is set to
  "dual", the "output" part of the pair is actually itself a pair of
  (autoencoder output, predictor output) graph nodes.
  """
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

def compile_model(params, input, output, mode):
  """
  Compiles the given model according to the mode (either "autoencoder" or
  "predictor").
  """
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

def get_encoding_model(auto_model, params):
  """
  Gets an encoding model from a trained autoencoder by pulling out the final
  layer according to its name.
  """
  return Model(
    inputs=auto_model.input,
    outputs=auto_model.get_layer(params["network"]["final_layer_name"]).output
  )

def load_images_into_items(items, params):
  """
  Takes data items and loads image data from the listed filenames into a new
  "image" column. Also computes the mean image and image deviation for each
  image.
  """
  # TODO: Resize images as they're loaded?
  all_images = []
  for i, filename in enumerate(items["file"]):
    utils.prbar(i / items["count"], debug=debug)
    img = imread(filename)
    img = img_as_float(img) # convert to [0, 1] floating-point range
    img = img[:,:,:3] # throw away alpha channel
    img = convert_colorspace(
      img,
      params["network"]["initial_colorspace"],
      params["network"]["training_colorspace"]
    )
    all_images.append(img)

  debug() # done with progress bar

  items["image"] = np.asarray(all_images)
  items["mean_image"] = np.mean(items["image"], axis=0)
  items["image_deviation"] = items["image"] - items["mean_image"]

def create_simple_generator(params):
  """
  Creates a simple image data generator from the input image directory listed
  in the given parameters.
  """
  return ImageDataGenerator().flow_from_directory(
    params["input"]["img_dir"],
    target_size=params["input"]["image_shape"][:-1],
    batch_size=1,
    shuffle=False,
    class_mode='sparse' # classes as integers
  )

def create_training_generator(items, params, mode="autoencoder"):
  """
  Creates a image data generator for training data using the input image
  directory listed in the given parameters.
  """
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
    debug("Invalid mode '{}'! Aborting.".format(mode))
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


def train_model(params, model, training_gen, n):
  """
  Trains the given model using the given training data generator for the given
  number of epochs. Returns nothing (just alters the weights of the given
  model).
  """
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

def rate_image(image, model):
  """
  Returns the rating for an individual image, which is just the model's error
  when attempting to reconstruct it.
  """
  as_batch = image.reshape((1,) + image.shape) # pretend it's a batch
  return model.test_on_batch(as_batch, as_batch)

def rate_images(items, model, params):
  """
  For each image, computes its reconstruction error under the given
  (autoencoder) model. Stores those values as a new "rating" column.
  """
  src = items["image"]
  if params["network"]["subtract_mean"]:
    src = items["image_deviation"]

  result = []
  debug("There are {} example images.".format(items["count"]))
  progress = 0
  for i, img in enumerate(src):
    utils.prbar(i / items["count"], debug=debug)
    result.append(rate_image(img, model))

  debug() # done with the progress bar
  return np.asarray(result)

def check_ratings(items, model):
  for i in range(len(items["image"])):
    sr = items["rating"][i]
    img = items["image"][i]
    tr = rate_image(img, model)
    if abs(sr - tr) > 0.0000001:
      debug("Unequal ratings found:", sr, tr)
      exit(1)

def spot_check_typicality(items, spots):
  for i, img, typ in spots:
    if not np.equal(items["image"][i], img).all():
      debug(
        "Error: typicality spot check found wrong image at index {}.".format(
          i
        )
      )
      exit(1)
    if items["typicality"][i] != typ:
      debug(
        "Error: typicality spot check found wrong value at index {}.".format(
          i
        )
      )
      exit(1)

def get_images(simple_gen):
  """
  Gets images and classes (automatically assigned based on subdirectories of
  the main target by ImageDataGenerator) from a simple image generator.
  """
  images = []
  classes = []
  debug("There are {} example images.".format(items["count"]))

  for i in range(items["count"]):
    img, cls = next(simple_gen)
    images.append(img[0])
    classes.append(cls[0])

  return images, classes

def images_sorted_by(items, sort_on):
  """
  Returns a list of all images sorted by the given column.
  """
  return np.asarray([
    pair[0] for pair in
      sorted(
        list(
          zip(items["image"], items[sort_on])
        ),
        key=lambda pair: pair[1]
      )
  ])

def save_images(images, params, directory, name_template, labels=None):
  """
  Saves the given images into the given directory, putting integer labels into
  the given name template. If a list of labels is given, they will be added as
  text inside the saved images using `mogrify -label`.
  """
  for i in range(len(images)):
    l = str(labels[i]) if (not (labels is None)) else None
    if l:
      img = impr.labeled(images[i], l)
    else:
      img = images[i]
    img = convert_colorspace(
      img,
      params["network"]["training_colorspace"],
      params["network"]["initial_colorspace"]
    )
    fn = os.path.join(
      params["output"]["directory"],
      directory,
      name_template.format("{:03}".format(i))
    )
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      imsave(fn, img)

def save_image_lineup(
  items,
  model,
  l, w,
  according_to,
  filename,
  params,
  show_means=None
):
  """
  Orders images according to the given column, and produces a lineup of l
  percentiles, sampling w images at random from each percentile to make an lxw
  montage. The images are merged together into a single image and saved. Note
  that some percentiles may be empty/deficient if the distribution of the
  according_to column is lumpy. If "show_means" is given, the mean of each
  percentile is added to the lineup at the bottom. If it is given as
  "deviation", instead the difference between the percentile mean and the
  overall mean is used. If "ampdiff" is given for "show_means", differences
  from the overall mean are amplified.
  """
  percentiles = np.linspace(0, 100, l)

  cutoffs = [
    np.percentile(items[according_to], prc, axis=0, interpolation="linear")
      for prc in percentiles
  ]

  stripes = []

  for i in range(l-1):
    indices = (
      (items[according_to] >= cutoffs[i])
    & (items[according_to] < cutoffs[i+1])
    )

    hits = items["image"][indices]
    if len(hits) > w:
      reps = hits.take(
        np.random.choice(len(hits), size=w, replace=False),
        axis=0
      )
    else:
      reps = hits

    if len(reps) > 0:
      stripe = impr.join([impr.frame(r) for r in reps], vert=True)
      if show_means == "deviation":
        dev = np.mean(hits, axis=0) - items["mean_image"]
        # per-channel boosts
        dev[:,:,0] *= 8
        dev[:,:,1] = 0.8
        dev[:,:,2] *= 8

        # per-channel normalization
        #dev[:,:,0] /= np.max(np.abs(dev[:,:,0]))
        #dev[:,:,0] /= np.max(np.abs(dev[:,:,0]))
        #dev[:,:,1] /= np.max(np.abs(dev[:,:,1]))
        #dev[:,:,2] /= np.max(np.abs(dev[:,:,2]))

        # shrink/wrap back from [-1,1] to [0,1]
        dev[:,:,0] = dev[:,:,0] % 1
        #dev[:,:,1] = 0.5 + dev[:,:,1] / 2
        dev[:,:,2] = 0.5 + dev[:,:,2] / 2
        dev[:,:,2] = dev[:,:,2].clip(0,1)

        stripe = impr.concatenate(
          stripe,
          impr.frame(dev),
          vert=True
        )
      elif show_means == "ampdiff":
        dev = np.mean(hits, axis=0) - items["mean_image"]
        # per-channel normalization
        dev[:,:,0] /= np.max(np.abs(dev[:,:,0]))
        dev[:,:,1] /= np.max(np.abs(dev[:,:,1]))
        dev[:,:,2] /= np.max(np.abs(dev[:,:,2]))

        diff = items["mean_image"] + dev

        diff[:,:,0] %= 1.0 # hue can just circle back
        diff[:,:,1] = diff[:,:,1].clip(0,1)
        diff[:,:,2] = diff[:,:,2].clip(0,1)

        stripe = impr.concatenate(
          stripe,
          impr.frame(diff),
          vert=True
        )
      elif show_means:
        stripe = impr.concatenate(
          stripe,
          impr.frame(np.mean(hits, axis=0)),
          vert=True
        )
    else:
      stripe = impr.frame(np.ones(params["input"]["image_shape"]))

    stripes.append(stripe)

  scale = impr.join(stripes, vert=False)

  img = convert_colorspace(
    scale,
    params["network"]["training_colorspace"],
    params["network"]["initial_colorspace"]
  )

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    imsave(os.path.join(params["output"]["directory"], filename), img)

def montage_images(params, directory, name_template, label=None):
  """
  Groups images in the given directory according to the given name template,
  after filling in a single '*'. Matching images are montages into a combined
  image, using "montage" for the name slot.
  """
  path = os.path.join(params["output"]["directory"], directory)
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

def collect_montages(params, directory, label_dirnames=False):
  """
  Collects all montages in the given directory (and any subdirectories,
  recursively) and groups them into one large combined montage. If
  label_dirnames is given, it labels each image with the name of the directory
  it was taken from.
  """
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


def get_features(images, model, params):
  """
  Given images and a model, returns the features of those images encoded with
  that model.
  """
  encoder = get_encoding_model(model, params)
  return encoder.predict(np.asarray(images))

def save_training_examples(items, generator, params):
  """
  Saves examples from the given training generator, including raw images,
  transformed images, and input images.
  """
  debug("  Saving training examples...")
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
    params,
    params["filenames"]["transformed_dir"],
    params["filenames"]["raw_example"]
  )
  save_images(
    ex_input,
    params,
    params["filenames"]["transformed_dir"],
    params["filenames"]["input_example"]
  )
  save_images(
    ex_output,
    params,
    params["filenames"]["transformed_dir"],
    params["filenames"]["output_example"]
  )
  montage_images(
    params,
    params["filenames"]["transformed_dir"],
    params["filenames"]["raw_example"]
  )
  montage_images(
    params,
    params["filenames"]["transformed_dir"],
    params["filenames"]["input_example"]
  )
  montage_images(
    params,
    params["filenames"]["transformed_dir"],
    params["filenames"]["output_example"]
  )
  collect_montages(params, params["filenames"]["transformed_dir"])
  debug("  ...done saving training examples...")

def analyze_duplicates(params, items):
  debug("  Analyzing duplicates...")
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

  # TODO: Why no duplicates?!?

  if (items["duplicates"] == 1).all():
    debug("  ...no duplicates found.")
    return

  reps = []
  skip = set()
  for i, img in enumerate(items["image"]):
    utils.prbar(i / items["count"], debug=debug)
    if i not in skip and items["duplicates"][i] >= 2:
      reps.append(i)
      for j in range(items["distances"].shape[1]):
        if items["distances"][i][j] == 0:
          skip.add(j)

  debug()

  representatives = items["image"][reps]
  duplications = items["duplicates"][reps]

  order = np.argsort(duplications)
  representatives = representatives[order]
  duplications = duplications[order]

  representatives = representatives[-params["output"]["large_pool_size"]:]
  duplications = duplications[-params["output"]["large_pool_size"]:]

  save_images(
    representatives,
    params,
    params["filenames"]["duplicates_dir"],
    params["filenames"]["duplicate_image"],
    labels=duplications
  )
  montage_images(
    params,
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

  debug("  ...done.")

def get_clusters(params, method, items, metric="euclidean"):
  results = {}
  debug("Clustering images using {}...".format(method.__name__))
  debug("  Using metric '{}'".format(metric))

  # We want only nearest-neighbors for novel clustering (otherwise sorting
  # edges is too expensive).
  if method == NovelClustering:
    (
      results["neighbor_distances"],
      results["nearest_neighbors"]
    ) = neighbors_from_distances(
      items["distances"],
      params["clustering"]["neighborhood_size"]
    )

  # If we're not using DBSCAN we don't need clustering_distance
  if method == DBSCAN:
    # Figure out what the distance value should be:
    debug("  Computing DBSCAN cutoff distance...")

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
    debug("   Distance sample:")
    debug(smp)
    debug("  ...done.")
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
      debug(
        "Error determining clustering distance: all values were zero!",
        file=sys.stderr
      )
      exit(1)

    debug(
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

  debug("  Clustering images...")
  result = None
  if method == NovelClustering:
    edges = get_neighbor_edges(
      results["neighbor_distances"],
      results["nearest_neighbors"]
    )
    results["clusters"] = method(
      items[params["clustering"]["input"]],
      edges=edges
    )
    results["cluster_assignments"] = cluster_assignments(
      items["count"],
      results["clusters"]
    )
  else:
    fit = model.fit(items[params["clustering"]["input"]])
    results["cluster_assignments"] = fit.labels_
    results["clusters"] = build_clustering(results["cluster_assignments"])
    results["cluster_assignments"] = cluster_assignments(results["clusters"])

  if method == DBSCAN:
    results["core_mask"]= np.zeros_like(fit.labels_, dtype=int)
    results["core_mask"][fit.core_sample_indices_] = 1
    core_count = np.count_nonzero(results["core_mask"])
    debug(
      "Core samples: {}/{} ({:.2f}%)".format(
        core_count,
        items["count"],
        100 * core_count / items["count"]
      )
    )
  debug("  ...done clustering.")

  results["cluster_ids"] = set(results["clusters"].keys())

  results["cluster_sizes"] = {
    k: results["clusters"][k]["size"]
      for k in results["clusters"]
  }

  if -1 in results["cluster_ids"]:
    outlier_count = sum(results["cluster_assignments"] == -1)
    debug(
      "  Found {} cluster(s) (with {:2.3f}% outliers)".format(
        len(results["clusters"]) - 1,
        100 * (outlier_count / items["count"])
      )
    )
  else:
    debug(
      "  Found {} cluster(s) (no outliers)".format(len(results["clusters"]))
    )

  if method == DBSCAN:
    return (
      results["clusters"],
      results["cluster_assignments"],
      results["cluster_ids"],
      results["cluster_sizes"],
      results["ordered_distances"],
      results["outer_distances"],
      results["core_mask"],
    )
  else:
    return (
      results["clusters"],
      results["cluster_assignments"],
      results["cluster_ids"],
      results["cluster_sizes"],
    )

@utils.twolevel_default_params(DEFAULT_PARAMETERS)
def analyze_dataset(**params):
  """
  Analyzes a TODO: HERE
  """
  # Seed random number generator (hopefully improve image loading times via
  # disk cache?)
  debug("Random seed is: {}".format(params["options"]["seed"]))
  random.seed(params["options"]["seed"])
  # Backup old output directory and create a new one:
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

  # First load the CSV data:
  items = load_data(params)

  debug('-'*80)
  debug("Loading {} images...".format(items["count"]))
  #simple_gen = create_simple_generator(params)
  #images, classes = get_images(simple_gen)
  #images, classes = load_images_into_items(items, params)
  load_images_into_items(items, params)
  if "mean_image" in params["analysis"]["methods"]:
    debug("  Saving mean image...")
    save_images(
      [items["mean_image"]],
      params,
      ".",
      params["filenames"]["mean_image"]
    )
  debug("  ...done loading images.")
  debug('-'*80)

  if params["options"]["mode"] == "detect":
    params["options"]["mode"] = utils.cached_value(
      lambda: "autoencoder",
      "mode",
      "str",
      debug=debug
    )
  else:
    debug("Selected mode '{}'".format(params["options"]["mode"]))
    utils.store_cache(params["options"]["mode"], "mode", "str")

  if params["options"]["mode"] == "dual":
    required_models = [
      "autoencoder-model",
      "predictor-model"
    ]
  else:
    required_models = [ params["options"]["mode"] + "-model" ]

  debug('-'*80)
  debug("Acquiring {} model...".format(params["options"]["mode"]))
  if params["options"]["mode"] == "dual":
    def get_models():
      nonlocal items, params
      debug("  Creating models...")
      inp, comp = setup_computation(items, params, mode="dual")
      ae_model = compile_model(params, inp, comp[0], mode="autoencoder")
      pr_model = compile_model(params, inp, comp[1], mode="predictor")
      debug("  ...done creating models.")
      debug("  Creating training generators...")
      ae_train_gen = create_training_generator(
        items,
        params,
        mode="autoencoder"
      )
      pr_train_gen = create_training_generator(
        items,
        params,
        mode="predictor"
      )
      debug("  ...done creating training generators.")
      if "training_examples" in params["analysis"]["methods"]:
        save_training_examples(items, ae_train_gen, params)
      debug("  Training models...")
      train_model(params, ae_model, ae_train_gen, items["count"])
      train_model(params, pr_model, pr_train_gen, items["count"])
      debug("  ...done training models.")
      return (ae_model, pr_model)

    ae_model, pr_model = utils.cached_values(
      get_models,
      ("autoencoder-model", "predictor-model"),
      ("h5", "h5"),
      override=params["options"]["model"],
      debug=debug
    )
  else:
    def get_model():
      nonlocal items, params
      debug("  Creating model...")
      inp, comp = setup_computation(
        items,
        params,
        mode=params["options"]["mode"]
      )
      model = compile_model(params, inp, comp, mode=params["options"]["mode"])
      debug("  ...done creating model.")
      debug("  Creating training generator...")

      train_gen = create_training_generator(
        items,
        params,
        mode=params["options"]["mode"]
      )
      debug("  ...done creating training generator.")
      if "training_examples" in params["analysis"]["methods"]:
        save_training_examples(items, train_gen, params)
      debug("  Training model...")
      if params["options"]["mode"] == "dual":
        train_model(params, ae_model, ae_train_gen, items["count"])
        train_model(params, pr_model, pr_train_gen, items["count"])
      else:
        train_model(params, model, train_gen, items["count"])
      debug("  ...done training model.")
      return model

    model = utils.cached_value(
      get_model,
      params["options"]["mode"] + "-model",
      typ="h5",
      override=params["options"]["model"],
      debug=debug
    )

  debug('-'*80)
  if params["options"]["mode"] == "dual":
    debug("Got models:")
    debug(ae_model.summary())
    debug("\n...and...\n")
    debug(pr_model.summary())
  else:
    debug("Got model:")
    debug(model.summary())
  debug('-'*80)

  if params["options"]["mode"] == "dual":
    # DEBUG:
    test_autoencoder(items, ae_model, params)
    test_predictor(items, pr_model, params)
  elif params["options"]["mode"] == "autoencoder":
    test_autoencoder(items, model, params)
  elif params["options"]["mode"] == "predictor":
    test_predictor(items, model, params)
  else:
    debug(
      "Error: Unknown mode '{}'. No tests to run.".format(
        params["options"]["mode"]
      ),
      file=sys.stderr
    )

def analyze_correlations(items, columns, against, params):
  """
  Analyzes correlations between a list of columns and a single alternate
  column. Produces reports in the output directory, including a combined
  report.
  """
  p_threshold = utils.sidak_alpha(
    params["analysis"]["confidence_baseline"],
    len(columns)
  )
  # TODO: Multiple-comparisons correction across all calls to this function and
  # other tests in the overall analysis!
  debug(
    "  ...using α={} for {} comparisons...".format(
      p_threshold,
      len(columns)
    )
  )
  utils.reset_color()
  for col in columns:
    other = items[col]

    r, p = pearsonr(items[against], other)
    if p < p_threshold:
      debug("  '{}': {}  (p={})".format(col, r, p))
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
        plt.ylim(-0.1, 1)
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
        #debug("  ...binning {} vs. {} for plotting...".format(col, against))
        #matrix, xe, ye = np.histogram2d(
        #  items[against],
        #  other,
        #  bins=resolution,
        #  normed=True
        #)
        #matrix = np.transpose(matrix)
        #debug("  ...done binning.")
        #debug("  ...plotting binned items...")
        #plt.matshow(
        #  matrix,
        #  fignum=0,
        #  cmap=plt.get_cmap("YlGnBu"),
        #  origin="lower"
        #)
        #debug("  ...done plotting.")
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
      debug("    '{}': not significant (p={})".format(col, p))

  montage_images(
    params,
    ".",
    params["filenames"]["correlation_report"].format(against, "{}")
  )

def distribution_type(items, col):
  """
  Makes a guess at a distribution of the data based purely on the column type.
  Returns either "binomial" or "normalish", which are used to make decisions
  about what kind of statistical analysis to conduct.

  TODO: Provide an explicit mapping from columns to their estimated
  distributions?
  """
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
  """
  Like distribution_type, but returns a statistic function instead of a string.
  The function should be applicable to the column of interest.
  """
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

def analyze_cluster_stats(
  items,
  clusters,
  which_stats,
  params,
  aname="all"
):
  """
  Analyzes the given set of parameters per-cluster, looking for clusters that
  differ from the general population for any of the target parameters. Produces
  reports in the output directory.
  """
  cstats = { c: {} for c in clusters }

  for c in cstats:
    indices = list(clusters[c]["vertices"])
    #indices = items["cluster_assignments"] == c
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
  debug(
    "  ...there are {} clusters above size {}...".format(
      nbig,
      params["clustering"]["significant_size"]
    )
  )

  # Compute desired confidence bounds:
  ntests = len(which_stats)
  total_tests = nbig * ntests

  # The Šidák correction:
  shared_alpha = utils.sidak_alpha(
    params["analysis"]["confidence_baseline"],
    total_tests
  )
  bootstrap_samples = max(10000, int(2/shared_alpha))
  debug(
    (
"  ...testing {} properties of {} clusters ({} total tests)...\n"
"  ...setting individual α={} for joint α={}...\n"
"  ...using {} samples for bootstrapping max(10000, 2/α)..."
    ).format(
      ntests, nbig, total_tests,
      shared_alpha, params["analysis"]["confidence_baseline"],
      bootstrap_samples
    )
  )

  debug("  ...extracting cluster stats...")
  for i, c in enumerate(big_enough):
    utils.prbar(i / len(big_enough), debug=debug)
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

  debug()
  debug("  ...done extracting cluster stats.")

  # Compute significant differences:
  small = { col: set() for col in which_stats }
  large = { col: set() for col in which_stats }
  diff = { col: set() for col in which_stats }
  overall_stats = {
    col: relevant_statistic(items, col)(items[col]) for col in which_stats
  }
  debug("  ...bootstrapping overall means...")
  overall_cis = {}
  for i, col in enumerate(which_stats):
    utils.prbar(i / len(which_stats), debug=debug)
    overall_cis[col] = confidence_interval(
      items[col],
      distribution_type(items, col),
      bound=shared_alpha,
      n_samples=bootstrap_samples
    )
  debug()
  debug("  ...done bootstrapping overall means.")

  debug("  ...computing cluster property significance...")
  for i, col in enumerate(which_stats):
    utils.prbar(i / len(which_stats), debug=debug)
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

  debug()
  debug("  ...done computing cluster property significance.")

  debug("  ...plotting & summarizing cluster stats...")
  for col in cinfo:
    if not cinfo[col]:
      continue

    with open(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["cluster_stats"].format(
          aname,
          "outliers"
        )[:-4] + ".txt"
      ),
      'a'
    ) as fout:
      if col in which_stats and any((small[col], large[col], diff[col])):
        debug("    ...{} is significant...".format(col))
        fout.write("Outliers for '{}':\n".format(col))
        fout.write(
          "  small: " + ", ".join(str(x) for x in sorted(list(small[col])))+"\n"
        )
        fout.write(
          "  large: " + ", ".join(str(x) for x in sorted(list(large[col])))+"\n"
        )
        fout.write(
          "  diff: " + ", ".join(str(x) for x in sorted(list(diff[col]))) + "\n"
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
    # label with cluster IDs:
    plt.xticks(x, [int(cid) for cid in cids])

    plt.title("{} ({} clusters)".format(col, nbig))
    plt.xlabel("cluster")
    if col in items["types"]:
      plt.ylabel(relevant_statistic(items, col).__name__)
    else:
      plt.ylabel(col)
    plt.savefig(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["cluster_stats"].format(aname, col)
      )
    )
  debug("  ...done plotting & summarizing.")

  montage_images(
    params,
    ".",
    params["filenames"]["cluster_stats"].format(aname, "{}")
  )

def reconstruct_image(items, img, model, params):
  """
  Given a trained autoencoder model, runs the given image through it and
  returns the reconstructed result.
  """
  img = img.reshape((1,) + img.shape) # pretend it's a batch
  pred = model.predict(img)[0]
  if params["network"]["subtract_mean"]:
    pred += items["mean_image"]
  return pred

def lineage_images(cluster):
  """
  Recursively assembles original and reconstructed lineage images for this
  cluster.
  """
  if "children" not in cluster or len(cluster["children"]) == 0:
    return cluster["rep_montage"], cluster["rec_montage"]
  else:
    cli = [lineage_images(c) for c in cluster["children"]]

    rep_li = [lipair[0] for lipair in cli]
    rec_li = [lipair[1] for lipair in cli]

    return (
      impr.join(
        [
          cluster["rep_montage"],
          impr.join([impr.frame(img) for img in rep_li]),
        ],
        vert=True
      ),
      impr.join(
        [
          cluster["rec_montage"],
          impr.join([impr.frame(img) for img in rec_li]),
        ],
        vert=True
      )
    )

def assemble_combined_cluster_images(clusters):
  """
  Assembles combined original and reconstructed images from the cluster
  montages in the given clusters. Builds a tree structure showing which
  clusters are inside which others.
  """
  roots = [cl for cl in clusters.values() if cl["parent"] == None]
  reps, recs = [], []

  for r in roots:
    rep_li, rec_li = lineage_images(r)
    reps.append(rep_li)
    recs.append(rec_li)

  return (
    impr.join([impr.frame(img) for img in reps]),
    impr.join([impr.frame(img) for img in recs])
  )

def test_autoencoder(items, model, params):
  """
  Given an autoencoder model, subjects it to various tests and analyses
  according to the analyze.methods parameter. This method is also responsible
  for adding ratings, features, and projections to the given data items.
  """
  items["rating"] = utils.cached_value(
    lambda: rate_images(items, model, params),
    "ratings",
    override=params["options"]["rank"],
    debug=debug
  )

  if params["analysis"]["double_check_ratings"]:
    debug("Checking fresh ratings...")
    check_ratings(items, model)

  debug('-'*80)
  items["norm_rating"] = items["rating"] / np.max(items["rating"])
  items["values"]["rating"] = "numeric"
  items["types"]["rating"] = "numeric"
  items["values"]["norm_rating"] = "numeric"
  items["types"]["norm_rating"] = "numeric"

  # features
  debug('-'*80)
  src = items["image"]
  if params["network"]["subtract_mean"]:
    src = items["image_deviation"]
  items["features"] = utils.cached_value(
    lambda: get_features(src, model, params),
    "features",
    override=params["options"]["features"],
    debug=debug
  )

  items["values"]["features"] = "numeric"
  items["types"]["features"] = "numeric"

  for i in range(len(items["features"][0])):
    items["feature-{}".format(i)] = items["features"][:,i]
    items["values"]["feature-{}".format(i)] = "numeric"
    items["types"]["feature-{}".format(i)] = "numeric"

  # projected
  debug('-'*80)
  tsne = TSNE(
    n_components=params["projection"]["target_dimensionality"],
    random_state=0,
    metric=params["projection"]["metric"]
  )
  items["projected"] = utils.cached_value(
    lambda: tsne.fit_transform(items["features"]),
    "projected",
    override=params["options"]["project"],
    debug=debug
  )

  items["values"]["projected"] = "numeric"
  items["types"]["projected"] = "numeric"

  # typicality
  debug('-'*80)
  items["typicality"] = utils.cached_value(
    lambda: typicality(
      items["features"],
      quiet=False,
      metric=params["clustering"]["metric"],
      significant_fraction=params["clustering"]["typicality_fraction"],
    ),
    "typicality",
    "pkl",
    override=params["options"]["typicality"],
    debug=debug
  )

  items["values"]["typicality"] = "numeric"
  items["types"]["typicality"] = "numeric"

  if params["analysis"]["spot_check_typicality"]:
    typ_spots = [
      (i, items["image"][i], items["typicality"][i])
        for i in np.random.choice(items["count"], 400, replace=False)
    ]
    spot_check_typicality(items, typ_spots)

  # Save the best images and their reconstructions:
  if "reconstructions" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Saving example images...")
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

    sorted_images = images_sorted_by(items, "rating")
    #sorted_images = images_sorted_by(items, "typicality")

    best = sorted_images[:params["output"]["example_pool_size"]]
    worst = sorted_images[-params["output"]["example_pool_size"]:]
    rnd = np.copy(items["image"])
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
      save_images(
        iset,
        params,
        params["filenames"]["examples_dir"],
        fnt
      )
      rec_images = []
      for img in iset:
        rec_images.append(reconstruct_image(items, img, model, params))
        save_images(
          rec_images,
          params,
          params["filenames"]["examples_dir"],
          "rec-" + fnt
        )

      montage_images(
        params,
        params["filenames"]["examples_dir"],
        fnt
      )
      montage_images(
        params,
        params["filenames"]["examples_dir"],
        "rec-" + fnt
      )
    collect_montages(params, params["filenames"]["examples_dir"])
    debug("  ...done.")

  if params["analysis"]["double_check_ratings"]:
    debug("Re-checking ratings...")
    check_ratings(items, model)

  # Get pairwise distances (used for clustering and more)
  debug("  Computing nearest-neighbor distances...")
  debug("  Using input '{}'".format(params["clustering"]["input"]))
  items["distances"] = pairwise.pairwise_distances(
    items[params["clustering"]["input"]],
    metric=params["clustering"]["metric"]
  )
  debug("  ...done.")

  debug('-'*80)
  if params["clustering"]["method"] == DBSCAN:
    (
      items["clusters"],
      items["cluster_assignments"],
      items["cluster_ids"],
      items["cluster_sizes"],
      items["ordered_distances"],
      items["outer_distances"],
      items["core_mask"],
    ) = utils.cached_values(
      lambda:
        get_clusters(
          params,
          params["clustering"]["method"],
          items,
          params["clustering"]["metric"]
        ),
      (
        "clusters",
        "cluster_assignments",
        "cluster_ids",
        "cluster_sizes",
        "ordered_distances",
        "outer_distances",
        "core_mask",
      ),
      ("pkl", "pkl", "pkl", "pkl", "pkl", "pkl", "pkl"),
      override=params["options"]["cluster"],
      debug=debug
    )
  else:
    (
      items["clusters"],
      items["cluster_assignments"],
      items["cluster_ids"],
      items["cluster_sizes"],
    ) = utils.cached_values(
      lambda: get_clusters(
        params,
        params["clustering"]["method"],
        items,
        params["clustering"]["metric"]
      ),
      ("clusters", "cluster_assignments", "cluster_ids", "cluster_sizes"),
      ("pkl", "pkl", "pkl", "pkl"),
      override=params["options"]["cluster"],
      debug=debug
    )

  debug("  ...done clustering images.")

  if "duplicates" in params["analysis"]["methods"]:
    debug('-'*80)
    analyze_duplicates(params, items)

  # isolation
  debug('-'*80)
  if "isolation" in list(items["clusters"].values())[0]:
    # derive directly from clusters if they have isolation information:
    items["isolation"] = utils.cached_value(
      lambda: derive_isolation(items["features"], items["clusters"]),
      "isolation",
      "pkl",
      override=params["options"]["isolation"],
      debug=debug
    )
  else:
    # otherwise re-cluster to get isolation information:
    nbd, nbi = neighbors_from_distances(
      pairwise.pairwise_distances(
        items[params["clustering"]["input"]],
        metric=params["clustering"]["metric"]
      ),
      params["clustering"]["neighborhood_size"]
    )
    edges = get_neighbor_edges(nbd, nbi)
    items["isolation"] = utils.cached_value(
      lambda: isolation(
        items["features"],
        edges=edges,
        quiet=False,
        metric=params["clustering"]["metric"]
      ),
      "isolation",
      "pkl",
      override=params["options"]["isolation"],
      debug=debug
    )

  items["values"]["isolation"] = "numeric"
  items["types"]["isolation"] = "numeric"

  if params["analysis"]["spot_check_typicality"]:
    spot_check_typicality(items, typ_spots)

  # Assemble image lineups:
  if "lineups" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Assembling image lineups...")
    for col in params["analysis"]["show_lineup"]:
      save_image_lineup(
        items,
        model,
        params["analysis"]["image_lineup_steps"],
        params["analysis"]["image_lineup_samples"],
        col,
        params["filenames"]["image_lineup"].format(col),
        params,
        show_means=True
        #show_means="deviation"
        #show_means="ampdiff"
      )
    debug("...done assembling lineups.")

  if params["analysis"]["spot_check_typicality"]:
    spot_check_typicality(items, typ_spots)

  # Plot a histogram of error values for all images:
  if "reconstruction_error" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Plotting reconstruction error histogram...")
    debug("  Error limits:", np.min(items["rating"]), np.max(items["rating"]))
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
    debug("  ...done.")

  if params["analysis"]["spot_check_typicality"]:
    spot_check_typicality(items, typ_spots)

  if "reconstruction_correlations" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Computing reconstruction correlations...")
    analyze_correlations(
      items,
      params["analysis"]["correlate_with_error"],
      "norm_rating",
      params
    )
    debug("  ...done.")

  if "typicality_correlations" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Analyzing typicality...")
    analyze_correlations(
      items,
      params["analysis"]["correlate_with_error"],
      "typicality",
      params
    )
    debug("  ...done.")

  if "isolation_correlations" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Analyzing isolation...")
    analyze_correlations(
      items,
      params["analysis"]["correlate_with_error"],
      "isolation",
      params
    )
    debug("  ...done.")

  if params["analysis"]["spot_check_typicality"]:
    spot_check_typicality(items, typ_spots)

  # Plot the t-SNE results:
  if "tSNE" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Plotting t-SNE results...")
    cycle = palettable.tableau.Tableau_20.mpl_colors
    colors = []
    alt_colors = []
    first_target = params["network"]["predict_target"][0]

    # subsample indices:
    if items["count"] > params["analysis"]["tsne_subsample"]:
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

    for cl in items["cluster_assignments"][ss]:
      if cl == -1:
        alt_colors.append((0.0, 0.0, 0.0)) # black
      else:
        alt_colors.append(cycle[cl % len(cycle)])

    if "typicality" in items:
      typ_colors = []
      cmap = plt.get_cmap("plasma")
      for t in items["typicality"][ss]:
        typ_colors.append(cmap(t))

    if "isolation" in items:
      iso_colors = []
      cmap = plt.get_cmap("plasma")
      for t in items["isolation"][ss]:
        iso_colors.append(cmap(t))

    axes = [(0, 1)]
    if items["projected"].shape[1] == 3:
      axes = [(0, 1), (0, 2), (1, 2)]

    for i, dims in enumerate(axes):
      utils.prbar(i / len(dims), debug=debug)
      # Plot using true colors:
      x, y = dims

      plt.clf()
      plt.scatter(
        items["projected"][ss,x],
        items["projected"][ss,y],
        s=0.25,
        c=colors
      )
      plt.title(first_target)
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
      plt.scatter(
        items["projected"][ss,x],
        items["projected"][ss,y],
        s=sizes,
        c=alt_colors
      )
      plt.title("clusters")
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
        plt.scatter(
          items["projected"][ss,x],
          items["projected"][ss,y],
          s=0.25,
          c=typ_colors
        )
        plt.title("typicality")
        plt.xlabel("t-SNE {}".format("xyz"[x]))
        plt.ylabel("t-SNE {}".format("xyz"[y]))
        plt.savefig(
          os.path.join(
            params["output"]["directory"],
            params["filenames"]["tsne"].format("typ", x, y)
          )
        )

      # Plot isolation:
      if "isolation" in items:
        plt.clf()
        plt.scatter(
          items["projected"][ss,x],
          items["projected"][ss,y],
          s=0.25,
          c=iso_colors
        )
        plt.title("isolation")
        plt.xlabel("t-SNE {}".format("xyz"[x]))
        plt.ylabel("t-SNE {}".format("xyz"[y]))
        plt.savefig(
          os.path.join(
            params["output"]["directory"],
            params["filenames"]["tsne"].format("iso", x, y)
          )
        )

    debug()
    # TODO: Less hacky here
    debug("  ...gathering TSNE images...")
    montage_images(
      params,
      ".",
      params["filenames"]["tsne"].format("*", "*", "{}")
    )
    debug("  ...done.")

  if params["analysis"]["double_check_ratings"]:
    debug("Re-checking ratings post-clustering...")
    check_ratings(items, model)

  if params["analysis"]["spot_check_typicality"]:
    spot_check_typicality(items, typ_spots)

  # Plot a histogram of pairwise distance values (if we computed them):
  if (
    "distance_histograms" in params["analysis"]["methods"]
and params["clustering"]["method"] == DBSCAN
  ):
    debug('-'*80)
    debug(
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
      plt.xlabel("Distance to {} Neighbor".format(utils.ordinal(col+1)))
      plt.ylabel("Number of Images")
      #plt.axis([0, 1.1*max(items["outer_distances"]), 0, 1.2 * max(n)])
      plt.savefig(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["histogram"].format("distance-{}".format(col))
        )
      )
      plt.clf()
      montage_images(
        params,
        ".",
        params["filenames"]["histogram"].format("distance-{}")
      )
    debug("  ...done.")

  if "nearest" in params["analysis"]["methods"]:
    try:
      os.mkdir(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["nearest_dir"]
        ),
        mode=0o755
      )
    except FileExistsError:
      pass

    nbd, nbi = neighbors_from_distances(items["distances"], 1)

    nbd = nbd.reshape((nbd.shape[0],))
    nbi = nbi.reshape((nbi.shape[0],))

    best = np.argsort(nbd)[:params["output"]["example_pool_size"]]

    rows = []
    for idx in best:
      fr = idx
      to = nbi[idx]
      d = nbd[idx]
      rows.append(
        impr.join(
          [
            impr.labeled(items["image"][fr], items["id"][fr]),
            impr.labeled(items["image"][to], items["id"][to]),
            (items["image"][fr] - items["image"][to]) == 0,
            impr.text_image("{:1.3f}".format(d))
          ]
        )
      )
    combined = impr.join(
      rows,
      vert=True
    )
    save_images(
      [combined],
      params,
      params["filenames"]["nearest_dir"],
      params["filenames"]["nearest"]
    )

  if "representatives" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Finding representatives...")
    try:
      os.mkdir(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["representatives_dir"]
        ),
        mode=0o755
      )
    except FileExistsError:
      pass

    items["representatives"] = find_representatives(
      items[params["clustering"]["input"]],
      distances=items["distances"]
    )

    rlist = sorted(
      list(items["representatives"].keys()),
      key=lambda r: len(items["representatives"][r])
    )

    debug("  ...found {} representatives...".format(len(rlist)))

    fn = params["filenames"]["representative"].format("overall", "{}")

    save_images(
      [
        impr.join(
          [
            np.mean(
              items["image"][list(items["representatives"][r]),:,:,:],
              axis=0
            ),
            items["image"][r,:,:,:]
          ]
        )
          for r in rlist
      ],
      params,
      params["filenames"]["representatives_dir"],
      fn,
      labels=[
        "{} ({})".format(
          i,
          len(items["representatives"][r])
        )
          for i, r in enumerate(rlist)
      ]
    )
    montage_images(
      params,
      params["filenames"]["representatives_dir"],
      fn,
      label=str("overall representatives")
    )

    items["rep_clusters"] = {
      i: {
        "size": len(items["representatives"][r]),
        "vertices": items["representatives"][r]
      }
        for i, r in enumerate(rlist)
    }

    debug("  ...done finding representatives.")

  if "representative_statistics" in params["analysis"]["methods"]:
    debug('-'*80)
    # Summarize statistics per-representative:
    debug("Summarizing representative statistics...")
    analyze_cluster_stats(
      items,
      items["rep_clusters"],
      params["analysis"]["analyze_per_representative"],
      params,
      aname="representative"
    )
    debug("  ...done.")

  if "exemplars" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Finding exemplars...")
    try:
      os.mkdir(
        os.path.join(
          params["output"]["directory"],
          params["filenames"]["exemplars_dir"]
        ),
        mode=0o755
      )
    except FileExistsError:
      pass

    for col in params["analysis"]["examine_exemplars"]:
      debug("  ...examining '{}'...".format(col))
      exs = find_exemplars(
        items[params["clustering"]["input"]],
        items[col],
        distances=items["distances"]
      )
      debug("  ...found exemplars; saving them...".format(col))

      # mapping from values to dirnames:
      vals = {}
      if items["types"][col] == "categorical":
        for value, val in sorted(list(items["values"][col].items())):
          vals[val] = "{}:{}".format(col, value)
      else:
        for val in sorted(set(items[col])):
          vals[val] = "{}:{}".format(col, val)

      for k in sorted(list(exs.keys())):
        try:
          os.mkdir(
            os.path.join(
              params["output"]["directory"],
              params["filenames"]["exemplars_dir"],
              vals[k]
            ),
            mode=0o755
          )
        except FileExistsError:
          pass

        thisdir = os.path.join(params["filenames"]["exemplars_dir"], vals[k])

        exemplars = exs[k]
        indices = [ex[0] for ex in exemplars]
        centralities = [ex[1] for ex in exemplars]
        separations = [ex[2] for ex in exemplars]

        fn = params["filenames"]["exemplar"].format(vals[k], "{}")
        save_images(
          items["image"][indices],
          params,
          thisdir,
          fn,
          labels=[
            "{} / {:.5f}".format(*x)
              for x in zip(centralities, separations)
          ]
        )
        montage_images(
          params,
          thisdir,
          fn,
          label=str(vals[k])
        )

    collect_montages(
      params,
      params["filenames"]["exemplars_dir"],
      label_dirnames=True
    )
    debug("  ...done finding exemplars.")

  if "distances" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Plotting distances...")
    sds = np.sort(
      items["distances"],
      axis=1
    )[:,params["clustering"]["dbscan_neighbors"]]
    plt.plot(sds)
    plt.xlabel("Index")
    plt.ylabel(
      "Distance to {} Neighbor".format(
        utils.ordinal(params["clustering"]["dbscan_neighbors"])
      )
    )
    plt.savefig(
      os.path.join(
        params["output"]["directory"],
        params["filenames"]["distances"].format(
          utils.ordinal(params["clustering"]["dbscan_neighbors"])
        )
      )
    )
    plt.clf()

    ods = np.sort(
      items["distances"],
      axis=1
    )[:,1:params["clustering"]["dbscan_neighbors"]+1]
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

    debug("  Number of nth-neighbor clones:")
    for k in sorted(list(skipped.keys())):
      debug("    {}: {}".format(k, skipped[k]))

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
    debug("  ...done.")

  if "cluster_sizes" in params["analysis"]["methods"]:
    debug('-'*80)
    # Plot cluster sizes
    debug("Plotting cluster sizes...")
    just_counts = sorted(list(items["cluster_sizes"].values()))
    plt.clf()
    plt.scatter(range(len(just_counts)), just_counts, s=2.5, c="k")
    plt.xlabel("cluster_assignments")
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
    debug("  ...done.")

  if "cluster_statistics" in params["analysis"]["methods"]:
    debug('-'*80)
    # Summarize statistics per-cluster:
    debug("Summarizing clustered statistics...")
    analyze_cluster_stats(
      items,
      items["clusters"],
      params["analysis"]["analyze_per_cluster"],
      params,
      aname="condensed"
    )
    debug("  ...done.")

  if "cluster_samples" in params["analysis"]["methods"]:
    # Show some of the clustering results (TODO: better):
    debug('-'*80)
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
        if v >= params["clustering"]["viz_size"]
    ])[-params["output"]["max_cluster_samples"]-1:]
    nvc = len(viz)
    debug("Sampling clustered images from {} clusters...".format(nvc))
    shuf = list(enumerate(items["image"]))
    random.shuffle(shuf)

    for i, c in enumerate(viz):
      utils.prbar(i / nvc, debug=debug)
      cluster_images = []
      rec_images = []
      for (i, img) in shuf:
        if i in items["clusters"][c]["vertices"]:
          rec = reconstruct_image(items, img, model, params)
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
          "cluster-{:04d}_({})".format(c, items["cluster_sizes"][c] + 1)
        )
        recdir = os.path.join(
          params["filenames"]["clusters_rec_dir"],
          "cluster-{:04d}_({})".format(c, items["cluster_sizes"][c] + 1)
        )
      items["clusters"][c]["rep_dir"] = thisdir
      items["clusters"][c]["rec_dir"] = recdir
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

      # Save to directories:
      save_images(
        cluster_images,
        params,
        thisdir,
        params["filenames"]["cluster_rep"]
      )
      save_images(
        rec_images,
        params,
        recdir,
        params["filenames"]["cluster_rep"]
      )
      montage_images(
        params,
        thisdir,
        params["filenames"]["cluster_rep"],
        label=str(items["clusters"][c]["size"])
      )
      montage_images(
        params,
        recdir,
        params["filenames"]["cluster_rep"],
        label=str(items["clusters"][c]["size"])
      )

      # Save to clusters:
      items["clusters"][c]["rep_montage"] = impr.labeled(
        impr.montage(cluster_images),
        "cluster {} ({} items)".format(c, items["clusters"][c]["size"])
      )
      items["clusters"][c]["rec_montage"] = impr.labeled(
        impr.montage(rec_images),
        "cluster {} ({} items)".format(c, items["clusters"][c]["size"])
      )

    debug() # done with the progress bar
    debug("  ...creating combined cluster sample image...")

    collect_montages(
      params,
      params["filenames"]["clusters_dir"],
      label_dirnames=True
    )
    collect_montages(
      params,
      params["filenames"]["clusters_rec_dir"],
      label_dirnames=True
    )

    # TODO: Lineage text!
    try:
      rep_montage, rec_montage = assemble_combined_cluster_images(
        items["clusters"]
      )
      save_images(
        [rep_montage, rec_montage],
        params,
        params["filenames"]["clusters_dir"],
        params["filenames"]["clusters_summary"],
        labels=["original", "reconstructed"]
      )
    except MemoryError:
      raise RuntimeWarning("Could not assemble lineage image in memory.")

    debug("  ...done.")

def test_predictor(items, model, params):
  """
  Like test_autoencoder, this method test and analyzes a model, in this case,
  it works with the predictor model instead of the autoencoder model. As with
  test_autoencoder, various analyses are enabled by adding strings to the
  analysis.methods parameter.
  """
  if "prediction_accuracy" in params["analysis"]["methods"]:
    debug('-'*80)
    debug("Analyzing prediction accuracy...")
    debug("  ...there are {} samples...".format(items["count"]))
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
          title=target.title(),
          debug=debug
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

    debug(" ...done.")


def plot_confusion_matrix(
  cm,
  classes,
  normalize=False,
  title='Confusion matrix',
  cmap=None,
  debug=print
):
  """
  Confusion matrix code from:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  cmap = cmap or plt.cm.Blues
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      debug("Normalized confusion matrix")
  else:
      debug('Confusion matrix, without normalization')

  debug(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

if __name__ == "__main__":
  global debug
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
    help="Recompute the model even if a cached value is found."
  )
  parser.add_argument(
    "-r",
    "--rank",
    action="store_true",
    help="Recompute rankings even if a cached value is found."
  )
  parser.add_argument(
    "-f",
    "--features",
    action="store_true",
    help="Recompute features even if a cached value is found."
  )
  parser.add_argument(
    "-j",
    "--project",
    action="store_true",
    help="Recompute t-SNE projection even if a cached value is found."
  )
  parser.add_argument(
    "-c",
    "--cluster",
    action="store_true",
    help="Recompute clusters even if a cached value is found."
  )
  parser.add_argument(
    "-t",
    "--typicality",
    action="store_true",
    help="Recompute typicality even if a cached value is found."
  )
  parser.add_argument(
    "-i",
    "--isolation",
    action="store_true",
    help="Recompute isolation even if a cached value is found."
  )
  parser.add_argument(
    "-F",
    "--fresh",
    action="store_true",
    help="Recompute everything. Equivalent to '-mrfjcti'."
  )
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
  if options.fresh:
    options.model = True
    options.rank = True
    options.features = True
    options.project = True
    options.cluster = True
    options.typicality = True
    options.isolation = True
    # Explicitly disable all caching
    utils.toggle_caching(False)

  debug = utils.get_debug(options.quiet)

  import_libraries()
  analyze_dataset(options=vars(options))
  #utils.run_strict(analyze_dataset, options=vars(options))

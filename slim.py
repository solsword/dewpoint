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
of regular dense layers that decrease by powers of two. Based on cnn.py but set
up to run without loading all of the data at once.
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
import multiprocessing
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
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1

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

EMOTION_COLUMNS = [
  "Grief_Sadness_Pensiveness_Neutral_Serenity_Joy_Ecstasy",
  "Loathing_Disgust_Boredom_Neutral_Acceptance_Trust_Admiration",
  "Rage_Anger_Annoyance_Neutral_Apprehension_Fear_Terror",
  "Vigilance_Anticipation_Interest_Neutral_Distraction_Surprise_Amazement"
]

EMOTION_SCALES = [
  "Grief-Ecstasy",
  "Loathing-Admiration",
  "Rage-Terror",
  "Vigilance-Amazement",
]

PERSONALITY_SCALES = [
  "Anxious:nervous:worrying_At-ease:calm:relaxed",
  "Friendly:warm:affectionate_Cold:aloof:reserved",
  "Imaginative:a-dreamer_Practical:down-to-earth",
  "Trusting:gullible:naive_Suspicious:skeptical:cynical",
  "Capable:efficient:competent_Inept:unprepared",
  "Even-tempered:easy-going_Irritable:angry:touchy",
  "Solitary:shy:avoids-crowds_Gregarious:sociable:outgoing",
  "Unartistic:uninterested-in-art_Sensitive-to-art-and-beauty",
  "Crafty:sly:manipulative_Frank:sincere:straightforward",
  "Disorganized:sloppy_Organized:neat:methodical",
  "Depressed:sad:pessimistic_Contented:optimistic",
  "Assertive:forceful:dominant_Submissive:a-follower",
  "Emotionally-sensitive:passionate_Unfeeling:unempathic",
  "Generous:giving:considerate_Selfish:stingy:greedy",
  "Dutiful:scrupulous_Unreliable:undependable",
  "Poised:comfortable-with-others_Self-conscious:awkward:timid",
  "Slow:lethargic:unenergetic_Active:vigorous:busy",
  "Habit-bound:prefers-routine_Innovative:prefers-variety",
  "Aggressive:competitive:stubborn_Compliant:cooperative:docile",
  "Lazy:unambitious:aimless_Ambitious:workaholic",
  "Impulsive:yielding-to-temptation_Controlled:self-restrained",
  "Adventurous:fun-loving:risk-taking_Avoids-excitement:stimulation",
  "Intellectually-curious:open-minded_Narrow-interests:bored-by-ideas",
  "Modest:humble:self-effacing_Arrogant:conceited",
  "Disciplined:persistent:strong-willed_Procrastinating:quitting:weak",
  "Resilient:copes-well-with-crises_Vulnerable:fragile:helpless",
  "Somber:dull:sober_Happy:cheerful:joyous",
  "Dogmatic:traditional:conservative_Liberal:free-thinking",
  "Ruthless:hard-headed:unsentimental_Sympathetic:humanitarian",
  "Spontaneous:careless:thoughtless_Cautious:reflective:careful"
]


#--------------------#
# Default Parameters #
#--------------------#

DEFAULT_PARAMETERS = {
  "options": {
    "mode": "autoencoder",
    "model": False,
    "reconstruct": False,
    "features": False,
    "fresh": False,
    "quiet": True,
    "seed": 23,
  },

  "input": {
    #"img_dir": os.path.join("data", "original"),
    "img_dir": os.path.join("data", "mii_flat"),
    "csv_file": os.path.join("data", "csv", "miiverse_profiles.clean.csv"),
    "csv_index_col": 2,

    "emotions_csv": os.path.join("data", "csv", "emotion_ratings.csv"),
    "emotions_index_col": "mii_id",
    "emotions_neutral_index": 4,
    "personalities_csv": os.path.join("data", "csv", "personality_traits.csv"),
    "personalities_index_col": "mii_id",
    "personalities_scale_size": 5,

    "id_template": re.compile(r"([^_]+)_([^_]+)_.*"), # Matches IDs in filenames
    "image_shape": (48, 48, 3), # target image shape
    "initial_colorspace": "RGB", # colorspace of input images
    "training_colorspace": "HSV", # colorspace to use for training
  },

  "data_processing": {
    "multi_field_separator": '|',
    "multi_columns": ["genres"],
    "categorical_columns": ["country_code", "competence"],
    "normalize_columns": [],
    "log_transform_columns": [
      "friends", "following", "followers", "posts", "yeahs"
    ],
    "binarize_columns": {
      "friends": { -1: "private", 0: "no_friends" },
      "competence": "auto",
      "country_code": "auto",
    },
    #"filter_on": [ "country_code(US)|country_code(JP)" ],
    #"filter_on": [ "private", "country_code(US)|country_code(JP)" ]
    "filter_on": [
      "!private",
      "country_code(US)|country_code(JP)"
    ],
    #"filter_on": [
    #  "!private",
    #  "!no-friends",
    #  "country_code(US)|country_code(JP)"
    #],

    "subset_size": np.inf,
    #"subset_size": 12000,
    #"subset_size": 10000,
    #"subset_size": 6000,
  },

  "network": {
    # training parameters
    "batch_size": 32,
    "percent_per_epoch": 1.0, # how much of the data to feed per epoch
    #"epochs": 200,
    "epochs": 100,
    #"epochs": 50,
    #"epochs": 10,
    #"epochs": 4,
    #"epochs": 1,

    # network layer sizes:
    "conv_sizes": [32, 16],
    "base_flat_size": 512,
    "feature_size": 128,

    # training functions
    "ae_loss_function": "mean_squared_error",
    "pr_loss_function": "binary_crossentropy",

    # network design choices:
    "activation": "relu",
    "sparsen": True, #whether or not to force sparse activation in dense layers
    "normalize_activation": False, # whether to add normalizing layers or not
    "regularization_coefficient": 1e-5, #how much l1 norm to add to the loss

    # prediction parameters
    "predict_targets": ["country_code(US)"],

    "final_layer_name": "final_layer",
  },

  # statistical analysis parameters:
  "analysis": {
    "core_feature": "features",

    "double_check_REs": True,

    "confidence_baseline": 0.05, # base desired confidence across all tests

    "correlation_plot_bins": 50,

    "correlate_with_error": [
      "country_code(US)",
      "competence(Beginner)",
      "competence(Expert)",
      "competence(Intermediate)",
      "log_friends",
      "log_following",
      "log_followers",
      "log_posts",
      "log_yeahs",
      #"no-friends", #can't correlate if it's being filtered
      "genres()",
    ] + [
      "genres({})".format(g) for g in ALL_GENRES
    ],

    "correlate_with_features": [
      "country_code(US)",
      "competence(Beginner)",
      "competence(Expert)",
      "competence(Intermediate)",
      "log_friends",
      "log_following",
      "log_followers",
      "log_posts",
      "log_yeahs",
      #"no-friends", #can't correlate if it's being filtered
      "genres()",
    ] + [
      "genres({})".format(g) for g in ALL_GENRES
    ],

    "analyze_per_representative": [
      "novelty",
      "country_code(US0",
      "competence(Beginner)",
      "competence(Expert)",
      "competence(Intermediate)",
      "log_friends",
      "log_following",
      "log_followers",
      "log_posts",
      "log_yeahs",
      "genres()",
    ] + [
      "genres({})".format(g) for g in ALL_GENRES
    ],

    "examine_exemplars": [
      "genres(Sports)",
      "genres(Music)",
      "country_code(US)",
      "competence"
    ],

    "predict_analysis": [ "confusion" ],

    "max_monotony": 0.975,

    "image_lineup_steps": 25,
    "image_lineup_samples": 6,
    #"image_lineup_samples": 16,
    "image_lineup_mean_samples": 2500,
    "feature_lineup_mean_samples": 500,
    "show_lineup": [
      "novelty",
      "log_posts",
    ],
    "lineup_features": True,
    #"lineup_features": False,
  },

  "output": {
    "directory": "out-slim",
    "history_name": "out-slim-hist-{}.zip",
    "keep_history": 4,
    "example_pool_size": 16,
    "large_pool_size": 64,
    "figure_font": "DejaVu Sans",
    "figure_text_size": {
      "title": 20,
      "labels": 18,
      "general": 16
    },
    "variable_marker_base_size": 1,
    "variable_marker_var_size": 19,
    "max_cluster_samples": 10000,
    "samples_per_cluster": 16,
  },

  "filenames": {
    "best_image": "A-best-image-{}.png",
    "sampled_image": "A-sampled-image-{}.png",
    "worst_image": "A-worst-image-{}.png",
    "image_lineup": "image-lineup-{}.png",

    "correlation_report": "correlation-{}-{}.pdf",
    "histogram": "histogram-{}.pdf",
    "cluster_stats": "{}-cluster-stats-{}.pdf",
    "analysis": "analysis-{}.pdf",
    "exemplar": "exemplar-{}-{}.png",
    "representative": "representative-{}-{}.png",

    "examples_dir": "examples",
    "exemplars_dir": "exemplars",
    "representatives_dir": "representatives",
  },
}

def simple_proportion(data):
  """
  Function for getting the proportion of a binary data column that's True. Can
  be used with bootstrapping to get confidence intervals for the true
  proportion.
  """
  return np.sum(data) / len(data)

def resolve_holm_bonferroni(tests, family_alpha=0.05):
  """
  Takes a list of tests, which are quintuples of p-value, success function,
  success arguments, failure function, failure arguments. For each test which
  rejects the null hypothesis under Holm-Bonferroni correction for the given
  family-wide alpha value (default 5%) the success function is run and given
  the success arguments, and for all other tests the failure function is run
  with the failure arguments.

  See: https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
  """
  st = sorted(tests, key=lambda triple: triple[0])
  reject = True
  m = len(tests)
  debug(
    (
      "...resolving {} tests at the {:.1f}% confidence level using the "
      "Holm-Bonferroni correction..."
    ).format(m, 100 * family_alpha)
  )
  successes = 0
  for k, (p, sf, sfargs, ff, ffargs) in enumerate(st):
    alpha = family_alpha / (m - k) # note +1 cancels with offset-by-one in k
    if reject:
      debug("  ...reached α={:.8f} with p={:.8f}...".format(alpha, p))
      if p > alpha:
        ff(*ffargs)
        reject = False
      else:
        successes += 1
        sf(*sfargs)
      # test confidence here
    else:
      debug("  ...failed α={:.8f} with p={:.8f}...".format(alpha, p))
      ff(*ffargs)
  debug(
    "  ...out of {} tests, {} rejected the null hypothesis...".format(
      len(st),
      successes
    )
  )

ALL_HB_TESTS = []
ALL_HB_CLEANUP = []

def register_stattest(p, sf, sfargs, ff, ffargs):
  """
  Registers a test (p value, success function, success args, failure function,
  and failure args) in the global registry for Holm-Bonferroni correction.
  """
  global ALL_HB_TESTS
  ALL_HB_TESTS.append((p, sf, sfargs, ff, ffargs))

def register_statcleanup(f, fargs):
  """
  Registers a function + arguments to be called after all stats have been
  checked.
  """
  global ALL_HB_CLEANUP
  ALL_HB_CLEANUP.append((f, fargs))

def check_stattests(family_alpha=0.05):
  """
  Resolves all registered tests, applying Holm-Bonferroni correction to achieve
  the desired family-wide error rate. Then calls all registered cleanup
  functions in the order of registration.
  """
  global ALL_HB_TESTS, ALL_HB_CLEANUP
  debug("Resolving statistical tests...")
  resolve_holm_bonferroni(ALL_HB_TESTS, family_alpha)
  debug("...calling stats cleanup functions...")
  for f, args in ALL_HB_CLEANUP:
    f(*args)
  debug("...done with statistics.")

def load_data(params):
  """
  Loads the data from the designated input file(s). Returns both a full dataset
  and a filtered + subsetted set.
  """
  debug("Loading data...")
  # read the csv file into a data frame:
  df = pd.read_csv(
    params["input"]["csv_file"],
    sep=',',
    header=0,
    index_col=params["input"]["csv_index_col"]
  )
  debug("  ...read main CSV file; loading supplementary files...")

  # read emotions data
  em = pd.read_csv(
    params["input"]["emotions_csv"],
    sep=',',
    header=0
  )

  # read personalities data
  pt = pd.read_csv(
    params["input"]["personalities_csv"],
    sep=',',
    header=0
  )

  # add empty emotion columns:
  for col, scale in zip(EMOTION_COLUMNS, EMOTION_SCALES):
    ranks = ["Not Applicable"] + col.split('_')
    neutral_name = scale + "-neutral"
    median_name = scale + "-median"

    df[scale] = np.nan
    df[neutral_name] = np.nan
    df[median_name] = np.nan

  # add empty personality columns:
  for col in PERSONALITY_SCALES:
    ends = col.split('_')
    reps = [ e.split(':')[0] for e in ends ]
    scale_name = '::'.join(reps)
    neutral_name = scale_name + "-neutral"
    median_name = scale_name + "-median"

    df[scale_name] = np.nan
    df[neutral_name] = np.nan
    df[median_name] = np.nan

  df["personality"] = np.nan

  # add emotion/personality info to each item for which we have data:
  for idx in df.index:
    em_hits = em[params["input"]["emotions_index_col"]] == idx
    if any(em_hits):
      for col, scale in zip(EMOTION_COLUMNS, EMOTION_SCALES):
        ranks = ["Not Applicable"] + col.split('_')
        neutral_name = scale + "-neutral"
        median_name = scale + "-median"

        which = em.loc[em_hits, col]

        values = np.array([ ranks.index(term) for term in which ])

        distr = np.array([ sum(values == i) for i in range(len(ranks)) ])

        if sum(values != 0):
          median = np.median(values[values != 0])
        else:
          median = np.nan

        neutral_proportion = sum(
          values == params["input"]["emotions_neutral_index"]
        ) / len(values)

        df.at[idx, scale] = distr
        df.at[idx, neutral_name] = neutral_proportion
        df.at[idx, median_name] = median

    pt_hits = pt[params["input"]["personalities_index_col"]] == idx
    if any(pt_hits):
      overall_personality = 0
      for col in PERSONALITY_SCALES:
        ends = col.split('_')
        reps = [ e.split(':')[0] for e in ends ]
        scale_name = '::'.join(reps)
        neutral_name = scale_name + "-neutral"
        median_name = scale_name + "-median"

        which = pt.loc(pt_hits, col)

        distr = np.array(
          [
            sum(which == i)
              for i in range(1, 1 + params["input"]["personalities_scale_size"])
          ]
        )

        medain = np.median(which)

        neutral_proportion = sum(
          which == (1 + (params["input"]["personalities_scale_size"] // 2))
        ) / len(which)

        overall_personality += 1 - neutral_proportion

        df.at[idx, scale_name] = distr
        df.at[idx, neutral_name] = neutral_proportion
        df.at[idx, median_name] = median

      overall_personality /= len(PERSONALITY_SCALES)
      df.at[idx, "personality"] = overall_personality

  debug("  ...read all CSV files; searching for image files...")
  df["image_file"] = ""
  # add image paths
  seen = 0
  for dp, dn, files in os.walk(params["input"]["img_dir"]):
    for f in files:
      if f.endswith(".jpg") or f.endswith(".png"):
        seen += 1
        utils.prbar(seen / len(df), debug=debug, interval=100)
        #debug("  ...'{}'...\r".format(f))
        fbase = os.path.splitext(os.path.basename(f))[0]
        match = params["input"]["id_template"].match(fbase)
        if not match:
          continue
        country = match.group(1) # TODO: Check this?
        idx = match.group(2)
        try:
          df.at[idx,"image_file"] = os.path.join(dp, f)
        except:
          debug("\nWarning: image '{}' has no entry in csv file.".format(idx))

  debug()

  debug("  ...found image files...")

  missing_images = df.loc[df.image_file == "",:]
  if len(missing_images):
    debug(
      "Note: Dropping {} entries missing image files.".format(
        len(missing_images)
      )
    )

  df = df.loc[df.image_file != "",:]

  debug("  ...found {} records..".format(len(df)))
  debug("  ...done loading data & finding images...")

  debug("Expanding multi-value columns...")
  for col in params["data_processing"]["multi_columns"]:
    debug("  ...expanding column '{}'...".format(col))
    df[col] = df[col].fillna(value="")
    series = df[col].map(
      lambda x: set(x.split(params["data_processing"]["multi_field_separator"]))
    )
    all_categories = sorted(
      list(set(itertools.chain.from_iterable(series.values)))
    )
    debug(
      "  Found {} categories for column '{}':".format(len(all_categories), col)
    )
    for c in all_categories:
      n = col + "({})".format(c)
      # TODO: Fix this to work for empty string?
      if n == col + "()":
        df[n] = df[col] == ""
      else:
        df[n] = df[col].str.contains(c)
      pr = sum(df[n]) / len(df)
      debug("    ({}) ~ {:.1f}%".format(c, pr * 100))

  debug("  ...done expanding multi-value columns...")

  debug("Converting field types & adding extended columns...")
  for col in params["data_processing"]["categorical_columns"]:
    df[col] = df[col].astype("category")

  for col in params["data_processing"]["normalize_columns"]:
    c = df[col]
    df[col + "_norm"] = (c - c.min()) / (c.max() - c.min())

  for col in params["data_processing"]["log_transform_columns"]:
    df["log_" + col] = np.log(df[col] + 0.5)

  bcs = params["data_processing"]["binarize_columns"]
  for col in bcs:
    if bcs[col] == "auto":
      if pdt.is_categorical_dtype(df[col].dtype):
        cats = df[col].cat.categories
      else:
        cats = set(df[col].values)

      for val in cats:
        df[col + "({})".format(val)] = df[col] == val
    else:
      for val in bcs[col]:
        df[bcs[col][val]] = df[col] == val

  debug("  ...done.")

  precount = len(df)

  # Filter data:
  filtered = df.copy()
  for fil in params["data_processing"]["filter_on"]:
    if fil[0] == "!":
      matches = fil[1:].split("|")
      mask = ~df.loc[:,matches].any(axis=1)
    else:
      matches = fil.split("|")
      mask = df.loc[:,matches].any(axis=1)
    filtered = filtered[mask]

  count = len(filtered)
  debug(
    "  Filtered {} rows down to {} accepted rows...".format(precount, count)
  )

  if count > params["data_processing"]["subset_size"]:
    debug(
      "  Subsetting from {} to {} accepted rows...".format(
        count,
        params["data_processing"]["subset_size"]
      )
    )
    ilist = list(range(count))
    count = params["data_processing"]["subset_size"]
    random.shuffle(ilist)
    ilist = ilist[:count]
    filtered = filtered.iloc[ilist,:]

    debug("  ...done.")

  return df, filtered

def setup_computation(params, mode="autoencoder"):
  """
  Sets up a neural network using Keras, and returns an (input, output) pair of
  computation graph nodes. If the mode is set to "dual", the "output" part of
  the pair is actually itself a pair of (autoencoder output, predictor output)
  graph nodes.
  """
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
    for t in params["network"]["predict_targets"]:
      outputs += 1
    predictions = Dense(outputs, activation=params["network"]["activation"])(x)
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
        activation=params["network"]["activation"],
      )(x)
      # TODO: dropout on the way back up?
      flat_size *= 2

    x = Dense(flattened_size, activation=params["network"]["activation"])(x)
    if params["network"]["normalize_activation"]:
      x = BatchNormalization()(x)

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

def fetch_image(params, data, idx):
  """
  Fetches the image at a given index.
  """
  fn = data.at[idx, "image_file"]
  if not fn:
    return None
  img = imread(fn)
  img = img_as_float(img) # convert to [0, 1] floating-point range
  img = img[:,:,:3] # throw away alpha channel
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

def image_iterator(params, data, batch_size=1):
  """
  Creates an iterator over image data based on the "image_file" column of the
  given data set. The iterator yields (index, image) pairs in batches of the
  given size. The iterator repeats indefinitely by cycling back to the
  beginning of the data when it runs out.
  """
  batch = []
  while True:
    for idx in data.index:
      img = fetch_image(params, data, idx)
      batch.append((idx, img))

      if len(batch) >= batch_size:
        yield batch
        batch = []

def create_training_generator(params, data, mode="autoencoder"):
  """
  Creates a image data generator for training data using the input image
  directory listed in the given parameters.
  """
  if mode == "autoencoder":
    def pairgen(datagen):
      while True:
        obatch = [img for (idx, img) in next(datagen)]
        obatch = np.asarray(obatch)
        yield (obatch, obatch)

  elif mode == "predictor":
    # TODO: Shuffle ordering?
    def pairgen(datagen):
      while True:
        batch = next(datagen)
        images = []
        labels = []
        for idx, img in batch:
          label = data.at[idx, params["network"]["predict_targets"]]
          images.append(img)
          labels.append(label)
        images = np.asarray(images)
        labels = np.asarray(labels)
        yield (images, labels)

  else:
    debug("Invalid mode '{}'! Aborting.".format(mode))
    exit(1)

  datagen = image_iterator(params, data, params["network"]["batch_size"])
  return pairgen(datagen)


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
  Returns the reconstruction_error for an individual image, which is just the
  RMSE between the image and its reconstruction (as opposed to the model's loss
  function, which includes an L1 regularization term).
  """
  rec = reconstruct_image(image, model)
  return np.sqrt(np.mean((image - rec)**2))

def compute_reconstruction_errors(params, data, model):
  """
  For each image, computes its reconstruction error under the given
  (autoencoder) model, returning a Series.
  """
  debug("Computing reconstruction errors...")
  datagen = image_iterator(params, data)
  result = pd.Series(name="reconstruction_error", index=data.index)
  debug("There are {} example images.".format(len(data)))
  progress = 0
  for idx in data.index:
    utils.prbar(progress / len(data), debug=debug)
    progress += 1
    iidx, img = next(datagen)[0]
    result.at[iidx] = rate_image(img, model)

  debug() # done with the progress bar
  debug("  ...done computing reconstruction errors.")
  return result

def spot_check_reconstruction_errors(params, data, model):
  for idx in np.random.choice(data.index, size=400, replace=False):
    sr = data.at[idx, "reconstruction_error"]
    img = fetch_image(params, data, idx)
    tr = rate_image(img, model)
    if abs(sr - tr) > 0.0000001:
      debug("Unequal REs found:", sr, tr)
      exit(1)

def save_image(params, image, filename):
  """
  Saves an individual image to the given (absolute) filename.
  """
  img = convert_colorspace(
    img,
    params["input"]["training_colorspace"],
    params["input"]["initial_colorspace"]
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    imsave(filename, img)

def save_images(params, images, directory, name_template, labels=None):
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
    fn = os.path.join(
      params["output"]["directory"],
      directory,
      name_template.format("{:03}".format(i))
    )
    save_image(params, img, fn)

def save_image_lineup(
  params,
  data,
  according_to,
  l, w,
  filename,
  show_means=False
):
  """
  Orders images according to the given column, and produces a lineup of l
  percentiles, sampling w images at random from each percentile to make an lxw
  montage. The images are merged together into a single image and saved. Note
  that some percentiles may be empty/deficient if the distribution of the
  according_to column is lumpy. If "show_means" is given it should be an
  integer, and the mean of each percentile will be added to the lineup at the
  bottom, using the given number of samples.
  """
  values = data[according_to]

  vcounts = values.value_counts()
  distinct = len(vcounts)

  monotony = vcounts.values[0] / len(data)

  if distinct == 1:
    debug(
      "Can't assemble lineup of feature '{}' with no variety.".format(
        according_to,
      )
    )
    return

  if monotony > params["analysis"]["max_monotony"]:
    debug(
      "Won't assemble lineup of boring feature '{}' (monotony={:.1f}%).".format(
        according_to,
        monotony * 100
      )
    )
    return

  nv = np.min(values[np.isfinite(values)])
  xv = np.max(values[np.isfinite(values)])

  # Note: tiebreaking behavior is arbitrary/unknown here
  order = np.argsort(values)

  bw = len(order) // l

  bins = []
  while len(order) > bw:
    bot = (values[order.iloc[0]] - nv) / (xv - nv)
    for i in range(bw, len(order)):
      new = values[order.iloc[i]]
      if new > bot:
        break
    b = order.iloc[:i]
    order = order[i:]
    bins.append(b)

  if len(order) < bw//2 and len(bins) > 1:
    # pile leftovers into the last bin instead of giving them their own bin
    bins[-1].append(order)
  else:
    # or create a new bin if there's enough leftovers and/or too few bins so
    # far
    bins.append(order)

  stripes = []

  debug(
    (
      "Assembling lineup for '{}' using {} bins...\n"
      "  ...there are {} distinct values and monotony is {:.1f}%..."
    )
    .format(
      according_to,
      len(bins),
      distinct,
      monotony * 100
    )
  )
  for i, b in enumerate(bins):
    # normalized extents values:
    bot = (values[b[0]] - nv) / (xv - nv)
    top = (values[b[-1]] - nv) / (xv - nv)

    utils.prbar(i / len(bins), debug=debug)
    debug(" [{}]".format(len(b)), end="")

    hits = data.index[b]
    if len(hits) > w:
      reps = hits.take(
        np.random.choice(len(hits), size=w, replace=False),
        axis=0
      )
    else:
      reps = hits

    if show_means:
      if len(hits) > show_means:
        mean_sample = hits.take(
          np.random.choice(len(hits), size=show_means, replace=False),
          axis=0
        )
      else:
        mean_sample = hits

    images = np.asarray([fetch_image(params, data, r) for r in reps])
    if len(images) > 0:
      stripe = impr.join([ impr.frame(img) for img in images ], vert=True)
      if show_means:
        msi = np.asarray([fetch_image(params, data, ms) for ms in mean_sample])
        stripe = impr.concatenate(
          stripe,
          impr.frame(np.mean(msi, axis=0)),
          vert=True
        )

      stripe = impr.labeled(stripe, "{:.2f}-{:.2f}".format(bot, top))
      stripe = impr.labeled(stripe, str(len(hits)))

    else:
      stripe = impr.frame(np.zeros(params["input"]["image_shape"]))

    stripes.append(stripe)

  debug()

  scale = impr.join(stripes, vert=False)

  img = convert_colorspace(
    scale,
    params["input"]["training_colorspace"],
    params["input"]["initial_colorspace"]
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

def get_features(params, data, model):
  """
  Given data and a model, returns the features of the images encoded with that
  model.
  """
  encoder = get_encoding_model(model, params)

  features = []
  batch = []
  for i, idx in enumerate(data.index):
    utils.prbar(i/len(data), debug=debug, interval=100)
    batch.append(fetch_image(params, data, idx))
    if len(batch) >= params["network"]["batch_size"]:
      features.extend(encoder.predict(np.asarray(batch)))
      batch = []

  # handle any leftovers:
  if batch:
    features.extend(encoder.predict(np.asarray(batch)))

  return pd.Series(features, index=data.index)

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
  data, filtered = load_data(params)

  debug('-'*80)

  if params["options"]["mode"] == "detect":
    params["options"]["mode"] = utils.cached_value(
      lambda: "autoencoder",
      "slim-mode",
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
      nonlocal filtered, params
      debug("  Creating models...")
      inp, comp = setup_computation(params, mode="dual")
      ae_model = compile_model(params, inp, comp[0], mode="autoencoder")
      pr_model = compile_model(params, inp, comp[1], mode="predictor")
      debug("  ...done creating models.")
      debug("  Creating training generators...")
      ae_train_gen = create_training_generator(
        params,
        filtered,
        mode="autoencoder"
      )
      pr_train_gen = create_training_generator(
        params,
        filtered,
        mode="predictor"
      )
      debug("  ...done creating training generators.")
      debug("  Training models...")
      train_model(params, ae_model, ae_train_gen, len(filtered))
      train_model(params, pr_model, pr_train_gen, len(filtered))
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
      nonlocal filtered, params
      debug("  Creating model...")
      inp, comp = setup_computation(
        params,
        mode=params["options"]["mode"]
      )
      model = compile_model(params, inp, comp, mode=params["options"]["mode"])
      debug("  ...done creating model.")
      debug("  Creating training generator...")

      train_gen = create_training_generator(
        params,
        filtered,
        mode=params["options"]["mode"]
      )
      debug("  ...done creating training generator.")

      debug("  Training model...")
      if params["options"]["mode"] == "dual":
        train_model(params, ae_model, ae_train_gen, len(filtered))
        train_model(params, pr_model, pr_train_gen, len(filtered))
      else:
        train_model(params, model, train_gen, len(filtered))
      debug("  ...done training model.")
      return model

    model = utils.cached_value(
      get_model,
      "slim-" + params["options"]["mode"] + "-model",
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

  test_autoencoder(params, data, filtered, model)


def plot_regression_line(ax, x, y, **style):
  # add a regression line:
  sx, ex = ax.get_xlim()
  lr_m, lr_b, lr_r, lr_p, lr_std = linregress(x, y)
  sy = lr_m * sx + lr_b
  ey = lr_m * ex + lr_b
  ax.add_line(
    Line2D(
      [sx, ex],
      [sy, ey],
      label=(
        style["label"]
          if "label" in style
          else "y = {:.6f} * x + {:.6f}".format(lr_m, lr_b)
      ),
      **style
    )
  )


def plot_proportion_histogram(params, data, ax, col, against, val=True):
  """
  Plots a histogram of proportions points where 'col' == 'val'  along the axis
  'against'.
  """
  x = np.linspace(
    np.min(data[against]),
    np.max(data[against]),
    params["analysis"]["correlation_plot_bins"]
  )

  baseline = sum(data[col]==val) / len(data)

  y = np.zeros((len(x)-1,), dtype=float) # array of binned proportions
  s = np.zeros((len(x)-1,), dtype=int) # array of bin counts
  for i in range(len(x)-1):
    if i == len(x)-2:
      # grab upper-extreme point:
      matches = (x[i] <= data[against]) & (data[against] <= x[i+1])
    else:
      matches = (x[i] <= data[against]) & (data[against] < x[i+1])
    total = sum(matches)
    s[i] = total
    if total != 0:
      y[i] = sum(data[col][matches]==val) / total
    else:
      #y[i] = -0.05 # nothing to count here; proportion is undefined
      y[i] = np.nan # nothing to count here; proportion is undefined

  ns = s / np.max(s)

  c = utils.pick_color()

  ax.axhline(baseline, lw=0.04, c=(0.7, 0.7, 0.7), label="base proportion")
  ax.scatter(
    x[:-1], y,
    c=c,
    s = (
      params["output"]["variable_marker_base_size"]
    + (ns * params["output"]["variable_marker_var_size"])
    ),
    label="binned proportions"
  )
  plot_regression_line(ax, data[against], data[col], lw=0.02, ls="dotted", c=c)
  ax.set_xlabel(against)
  if pdt.is_bool_dtype(data[col].dtype):
    ax.set_ylabel("{} proportion".format(col))
  else:
    ax.set_ylabel("{} = {} proportion".format(col, val))
  ax.set_ylim(-0.1, 1)
  ax.legend()


def plot_means_histogram(params, data, ax, col, against):
  """
  Plots a histogram of the means of 'col' values within bins of 'against'
  values.
  """
  x = np.linspace(
    np.min(data[against]),
    np.max(data[against]),
    params["analysis"]["correlation_plot_bins"]
  )

  baseline = np.mean(data[col])

  y = np.zeros((len(x)-1,), dtype=float) # array of binned proportions
  s = np.zeros((len(x)-1,), dtype=int) # array of bin counts

  for i in range(len(x)-1):
    if i == len(x)-2:
      # grab upper-extreme point:
      matches = (x[i] <= data[against]) & (data[against] <= x[i+1])
    else:
      matches = (x[i] <= data[against]) & (data[against] < x[i+1])
    total = sum(matches)
    s[i] = total
    if total != 0:
      y[i] = np.mean(data[col][matches])
    else:
      y[i] = np.nan # nothing to count here; mean is undefined
      #y[i] = -0.05 # nothing to count here; mean is undefined

  ns = s / np.max(s)

  c = utils.pick_color()

  ax.axhline(baseline, lw=0.04, c=(0.7, 0.7, 0.7), label="overall mean")
  ax.scatter(
    x[:-1], y,
    c=c,
    s=0.02 + 7.8*ns,
    label="binned means"
  )
  plot_regression_line(ax, data[against], data[col], lw=0.02, ls="dotted", c=c)
  ax.set_xlabel(against)
  ax.set_ylabel(col + " mean")

  dmin = np.min(data[col])
  dmax = np.max(data[col])
  drange = dmax - dmin
  ax.set_ylim(dmin - 0.1 * drange, dmax + 0.1 * drange)

  ax.legend()


def plot_contrasting_distributions(params, data, ax, col, against):
  """
  Creates a violin plot of the distributions of 'against' for each value of
  'col'.
  """
  ax.axhline(
    np.mean(data[against]),
    lw=0.04,
    c=(0.7, 0.7, 0.7),
    label="overall mean"
  )
  values = sorted(list(set(data[col].values)))
  #ax.boxplot(
  #  [
  #    data.loc[data[col]==v, against]
  #      for v in values
  #  ],
  #  notch=True,
  #  sym='',
  #  labels=["{} ({})".format(v, sum(data[col]==v)) for v in values]
  #)
  ax.violinplot(
    [
      data.loc[data[col]==v, against]
        for v in values
    ],
    showmeans=True
  )
  ax.set_xticks(range(1, len(values) + 1))
  ax.set_xticklabels(
    ["{} ({})".format(v, sum(data[col]==v)) for v in values]
  )

  ax.set_xlabel(col)
  ax.set_ylabel(against + " distribution")
  ax.legend()


def plot_correlation(params, data, ax, col, against):
  """
  Plots correlation between 'col' and 'against', using a different kind of plot
  depending on the underlying data type of 'col'.
  """
  vtype = data[col].dtype
  if pdt.is_bool_dtype(vtype):
    plot_proportion_histogram(params, data, ax, col, against)

  elif pdt.is_numeric_dtype(vtype):
    plot_means_histogram(params, data, ax, col, against)
  else:
    # give up and do a scatterplot
    ax.scatter(data[against], data[col], s=0.25)
    ax.set_xlabel(against)
    ax.set_ylabel(col)


def plot_rev_correlation(params, data, ax, col, against):
  vtype = data[col].dtype
  if pdt.is_bool_dtype(vtype) or pdt.is_categorical_dtype(vtype):
    plot_contrasting_distributions(params, data, ax, col, against)
  elif pdt.is_numeric_dtype(vtype):
    plot_means_histogram(params, data, ax, against, col)
  else:
    # give up and do a scatterplot
    debug(
      "Warning: don't know how to plot separation of type '{}'.".format(vtype)
    )
    ax.scatter(data[col], data[against], s=0.25)
    ax.set_xlabel(col)
    ax.set_ylabel(against)

def analyze_correlations(params, data, columns, against):
  """
  Analyzes correlations between a list of columns and a single alternate
  column. Produces reports in the output directory, including a combined
  report.
  """
  # TODO: Multiple-comparisons correction across all calls to this function and
  # other tests in the overall analysis!
  debug("  ...correlating against '{}'...".format(against))
  debug("  ...scheduling {} comparisons...".format(len(columns) * 2))
  utils.reset_color()
  for col in columns:
    vtype = data[col].dtype
    if pdt.is_bool_dtype(vtype):
      tn = "t"
      t, p = ttest_ind(
        data.loc[data[col]==True, against],
        data.loc[data[col]==False, against],
        equal_var=False, # Apply Welch's correction for non-equal variances
        nan_policy="omit" # shouldn't matter
      )
      es = (
        np.mean(data.loc[data[col]==True, against])
      - np.mean(data.loc[data[col]==False, against])
      )
    elif pdt.is_numeric_dtype(vtype):
      tn = "r"
      es, p = pearsonr(data[col], data[against])
    else:
      # TODO: HERE
      debug(
        "Warning: don't know how to test for '{}' column.".format(
          vtype
        )
      )
      tn = '?'
      es = -1
      p = 1.0

    box = [ None ]

    def sf(_box):
      _box[0] = True

    def ff(_box):
      _box[0] = False

    def clf(_box, _col, _against, _tn, _es, _p):
      save = False
      if _box[0]:
        debug(
          "'{}' vs '{}': {}={:.4f} (p={})".format(
            _col,
            _against,
            _tn,
            _es,
            _p
          )
        )
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_correlation(params, data, ax1, _col, _against)
        plot_rev_correlation(params, data, ax2, _col, _against)
        fig.set_size_inches(20, 9)
        fig.set_dpi(300)
        save = True
      else:
        debug(
          " -- '{}' vs '{}' FAILED (p={})".format(
            _col,
            _against,
            _p,
          )
        )

      if save:
        fig.savefig(
          os.path.join(
            params["output"]["directory"],
            params["filenames"]["correlation_report"].format(_against, _col)
          )
        )
        plt.close(fig)
        fig.clf()
        del fig

    register_stattest(
      p,
      sf, (box,),
      ff, (box,),
    )

    register_statcleanup(
      clf,
      (box, col, against, tn, es, p)
    )

  def cl_montage(against):
    montage_images(
      params,
      ".",
      params["filenames"]["correlation_report"].format(against, "{}")
    )

  register_statcleanup(cl_montage, (against,))

def distribution_type(data, col):
  """
  Makes a guess at a distribution of the data based purely on the column type.
  Returns either "binomial" or "normalish", which are used to make decisions
  about what kind of statistical analysis to conduct.

  TODO: Provide an explicit mapping from columns to their estimated
  distributions?
  """
  vtype = data[col].dtype
  if pdt.is_bool_dtype(vtype):
    return "binomial"
  elif pdt.is_categorical_dtype(vtype):
    if len(set(data[col].values)) == 2:
      return "binomial"
    else:
      raise RuntimeWarning(
"Warning: Using 'normalish' distribution for multi-category column '{}'."
        .format(col, vtype)
      )
      return "normalish"
  elif pdt.is_numeric_dtype(vtype):
    # TODO: actually test this here?
    return "normalish"
  else:
    raise RuntimeWarning(
"Warning: Using 'normalish' distribution for column '{}' with type '{}'."
      .format(col, vtype)
    )
    return "normalish"

def relevant_statistic(data, col):
  """
  Like distribution_type, but returns a statistic function instead of a string.
  The function should be applicable to the column of interest.
  """
  vtype = data[col].dtype

  if pdt.is_bool_dtype(vtype):
    stat = simple_proportion
  elif pdt.is_categorical_dtype(vtype):
    if len(set(data[col].values)) == 2:
      stat = simple_proportion
    else:
      stat = np.average
  elif pdt.is_numeric_dtype(vtype):
    stat = np.average
  else:
    stat = np.average
    raise RuntimeWarning(
"Warning: Using 'average' stat for column '{}' with type '{}'."
      .format(col, vtype)
    )
  return stat

def analyze_cluster_stats(
  params,
  data,
  clusters,
  which_stats,
  aname="all"
):
  """
  Analyzes the given set of parameters per-cluster, looking for clusters that
  differ from the general population for any of the target parameters. Produces
  reports in the output directory.
  """
  # TODO: How to do Holm-Bonferroni correction for this?!?
  cstats = { c: {} for c in clusters }

  for c in cstats:
    indices = list(clusters[c]["vertices"])
    #indices = data["cluster_assignments"] == c
    for col in which_stats:
      cstats[c][col] = data[col][indices]
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

      stat = relevant_statistic(data, col)
      dist = distribution_type(data, col)

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
    col: relevant_statistic(data, col)(data[col]) for col in which_stats
  }
  debug("  ...bootstrapping overall means...")
  overall_cis = {}
  for i, col in enumerate(which_stats):
    utils.prbar(i / len(which_stats), debug=debug)
    overall_cis[col] = confidence_interval(
      data[col],
      distribution_type(data, col),
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

      vtype = data[col].dtype
      if pdt.is_bool_dtype(vtype):
        # Use Fisher's exact test
        in_and_true = len([x for x in cstats[c][col] if x])
        in_and_false = cstats[c]["size"] - in_and_true

        # TODO: Exclude cluster members?
        all_and_true = sum(data[col].astype(bool))
        all_and_false = len(data) - all_and_true

        table = np.array(
          [
            [ in_and_true,  all_and_true  ],
            [ in_and_false, all_and_false ]
          ]
        )
        statname = "odds"
        stat, p = fisher_exact(table, alternative="two-sided")

      elif pdt.is_categorical_dtype(vtype):
        # Use Pearson's chi-squared test
        values = sorted(list(set(data[col].values)))
        # TODO: This normalization is *REALLY* sketchy and only vaguely related
        # to Laplace smoothing! Find another way of comparing things.
        nf = 1 / len(cstats[c][col])
        contingency = np.array(
          [
            [ nf + sum([v == val for v in cstats[c][col]]) for val in values ],
            [ nf + sum([v == val for v in data[col]]) for val in values ],
          ]
        )
        statname = "chi²"
        stat, p, dof, exp = chi2_contingency(contingency, correction=True)

      elif pdt.is_numeric_dtype(vtype):
        # TODO: Something else for power-distributed variables?
        # Note: log-transformation has been applied to some stats above
        # Note: may require distribution-fitting followed by model
        # log-likelihood ratio analysis.
        stat, p = ttest_ind(
          cstats[c][col],
          data[col],
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
            vtype
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
    plt.xticks(x, [int(cid) for cid in cids], size=5, rotation=90)

    plt.title("{} ({} clusters)".format(col, nbig))
    plt.xlabel("cluster")
    if col in data.columns:
      plt.ylabel(relevant_statistic(data, col).__name__)
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

def reconstruct_image(img, model):
  """
  Given a trained autoencoder model, runs the given image through it and
  returns the reconstructed result.
  """
  img = img.reshape((1,) + img.shape) # pretend it's a batch
  return model.predict(img)[0]

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

def test_autoencoder(params, data, filtered, model):
  """
  Given an autoencoder model, subjects it to various tests and analyses. This
  method is also responsible for adding reconstruction_errors and features to
  the given data.
  """
  debug('-'*80)

  data["reconstruction_error"] = utils.cached_value(
    lambda: compute_reconstruction_errors(params, data, model),
    "slim-reconstruction_errors",
    override=params["options"]["reconstruct"],
    debug=debug
  )

  filtered["reconstruction_error"] = data["reconstruction_error"]

  if params["analysis"]["double_check_REs"]:
    debug("Checking fresh reconstruction errors...")
    spot_check_reconstruction_errors(params, data, model)

  # normalize relative to filtered subset but compute for entire dataset
  nre = np.min(filtered["reconstruction_error"])
  xre = np.max(filtered["reconstruction_error"])
  data["novelty"] = (data["reconstruction_error"] - nre) / (xre - nre)

  filtered["novelty"] = data["novelty"]

  # features
  debug('-'*80)

  data["features"] = utils.cached_value(
    lambda: get_features(params, data, model),
    "slim-features",
    override=params["options"]["features"],
    debug=debug
  )

  filtered["features"] = data["features"]

  # break out individual features into their own columns:
  for i in range(params["network"]["feature_size"]):
    fn = "feature({})".format(i)
    data[fn] = data["features"].map(lambda f: f[i])
    filtered[fn] = data[fn]

  # Save the best images and their reconstructions:
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

  ordering = np.argsort(data["reconstruction_error"])

  best = ordering[:params["output"]["example_pool_size"]]
  worst = ordering[-params["output"]["example_pool_size"]:]
  rnd = list(range(len(data)))
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
    images = [ fetch_image(params, data, data.index[i]) for i in iset ]

    labels = [
      "{:.4f}".format(data.at[data.index[i], "reconstruction_error"])
        for i in iset
    ]

    montage = impr.montage(images, labels=labels, label_color=(1, 0, 1))

    rec_montage = impr.montage(
      [ reconstruct_image(img, model) for img in images ],
      labels=labels,
      label_color=(1, 0, 1)
    )

    save_image(params, montage, fnt.format("montage"))
    save_image(params, rec_montage, fnt.format("rec-montage"))

  # TODO: This using impr as well?
  collect_montages(params, params["filenames"]["examples_dir"])
  debug("  ...done.")

  if params["analysis"]["double_check_REs"]:
    debug("Re-checking reconstruction errors...")
    spot_check_reconstruction_errors(params, data, model)

  # Analyze feature variety:
  debug('-'*80)
  debug("Checking feature sparsity...")
  varieties = np.array([
    len(set(filtered["feature({})".format(i)].values))
      for i in range(params["network"]["feature_size"])
  ])
  monotonies = np.array([
    filtered["feature({})".format(i)].value_counts()[0]
      for i in range(params["network"]["feature_size"])
  ])
  novar = (varieties == 1)
  boring = (monotonies > (params["analysis"]["max_monotony"] * len(filtered)))
  drop = np.arange(len(varieties))[novar | boring]
  if len(drop) > 0:
    debug(
      "  ...found {} empty and {} boring features:".format(
        sum(novar),
        sum(boring)
      )
    )
    debug(
      "  ...{} useful features remain.".format(
        params["network"]["feature_size"] - len(drop)
      )
    )
  else:
    debug("  ...no empty features present...")

  debug("  ...assembling minified features...")
  active_features = []
  j = 0
  for i in range(params["network"]["feature_size"]):
    if i in drop:
      continue
    n = "mini_feature({})".format(j)
    # normalize the mini-features (relative to the filtered range)
    fn = "feature({})".format(i)
    fmin = np.min(filtered[fn])
    fmax = np.max(filtered[fn])
    data[n] = (data[fn] - fmin) / (fmax - fmin)
    filtered[n] = data[n]
    j += 1
    active_features.append(n)

  mini_features = data.loc[:,active_features].values
  data["mini_feature"] = None
  for i, idx in enumerate(data.index):
    utils.prbar(i / len(data), debug=debug, interval=100)
    data.at[idx, "mini_feature"] = mini_features[i]

  filtered["mini_feature"] = data["mini_feature"]

  debug()
  debug("  ...done checking feature sparsity.")

  debug('-'*80)
  debug(
    "Computing reconstruction correlations vs. {} filtered items...".format(
      len(filtered)
    )
  )
  analyze_correlations(
    params,
    filtered,
    params["analysis"]["correlate_with_error"],
    "novelty"
  )
  debug("  ...done.")

  debug('-'*80)
  has_emo = data[EMOTION_SCALES[0] + "-neutral"] != np.nan
  debug(
    "Computing emotion correlations vs. {} rated items...".format(sum(has_emo))
  )

  analyze_correlations(
    params,
    data[has_emo],
    [scale + "-neutral" for scale in EMOTION_SCALES],
    "novelty"
  )

  debug("  ...done.")

  debug('-'*80)
  has_per = data["personality"] != np.nan
  debug(
    "Computing personality correlations vs. {} rated items...".format(
      sum(has_per)
    )
  )

  analyze_correlations(
    params,
    data[has_per],
    ["personality"],
    "novelty"
  )

  debug("  ...done.")


  # TODO: Test feature correlations?
  #debug('-'*80)
  #debug("Computing feature correlations...")
  #for feature in active_features:
  #  analyze_correlations(
  #    params,
  #    data,
  #    params["analysis"]["correlate_with_features"],
  #    feature
  #  )
  #debug("  ...done.")


  # Assemble image lineups:
  debug('-'*80)
  debug("Assembling image lineups...")
  for col in params["analysis"]["show_lineup"]:
    save_image_lineup(
      params,
      data,
      col,
      params["analysis"]["image_lineup_steps"],
      params["analysis"]["image_lineup_samples"],
      params["filenames"]["image_lineup"].format(col),
      show_means=params["analysis"]["image_lineup_mean_samples"]
    )
  if params["analysis"]["lineup_features"]:
    for col in active_features:
      save_image_lineup(
        params,
        data,
        col,
        params["analysis"]["image_lineup_steps"],
        params["analysis"]["image_lineup_samples"],
        params["filenames"]["image_lineup"].format(col),
        show_means=params["analysis"]["feature_lineup_mean_samples"]
      )
  debug("...done assembling lineups.")

  # Plot a histogram of error values for all images:
  debug('-'*80)
  debug("Plotting reconstruction error histogram...")
  debug(
    "  Error limits:",
    np.min(data["reconstruction_error"]),
    np.max(data["reconstruction_error"])
  )
  plt.clf()
  n, bins, patches = plt.hist(data["reconstruction_error"], 100)
  #plt.plot(bins)
  plt.xlabel("Reconstruction Error")
  plt.ylabel("Number of Images")
  #plt.axis([0, 1.1*max(data["reconstruction_error"]), 0, 1.2 * max(n)])
  #plt.show()
  plt.savefig(
    os.path.join(
      params["output"]["directory"],
      params["filenames"]["histogram"].format("error")
    )
  )
  plt.clf()
  debug("  ...done.")

  if params["analysis"]["double_check_REs"]:
    debug("Re-checking reconstruction errors...")
    spot_check_reconstruction_errors(params, data, model)

  # TODO: DEBUG
  return

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

  data["representatives"] = find_representatives(
    data[params["analysis"]["core_feature"]],
    distances=data["distances"] # TODO: Really compute these?
  )

  rlist = sorted(
    list(data["representatives"].keys()),
    key=lambda r: len(data["representatives"][r])
  )

  total_represented = sum(len(v) for v in data["representatives"].values())

  debug(
    "  ...found {} representatives with {:.1f}% coverage...".format(
      len(rlist),
      100 * total_represented / len(data)
    )
  )

  fn = params["filenames"]["representative"].format("overall", "{}")

  save_images(
    params,
    [
      impr.join(
        [
          np.mean(
            [
              fetch_image(params, data, idx)
                for idx in list(data["representatives"][r])
            ],
            axis=0
          ),
          fetch_image(params, data, data.index[r])
        ]
      )
        for r in rlist
    ],
    params["filenames"]["representatives_dir"],
    fn,
    labels=[
      "{} ({})".format(
        i,
        len(data["representatives"][r])
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

  data["rep_clusters"] = {
    i: {
      "size": len(data["representatives"][r]),
      "vertices": data["representatives"][r],
      "edges": [ (r, to, 0, 0.25) for to in data["representatives"][r] ],
    }
      for i, r in enumerate(rlist)
  }

  debug("  ...done finding representatives.")

  if "representative_statistics" in params["analysis"]["methods"]:
    debug('-'*80)
    # Summarize statistics per-representative:
    debug("Summarizing representative statistics...")
    analyze_cluster_stats(
      params,
      data,
      data["rep_clusters"],
      params["analysis"]["analyze_per_representative"],
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
      #exs = find_exemplars(
      exs = find_alt_exemplars(
        data[params["analysis"]["core_feature"]],
        data[col],
        distances=data["distances"] # TODO: These?
      )
      debug("  ...found exemplars; saving them...".format(col))

      # mapping from values to dirnames:
      vals = {}
      vtype = data[col].dtype
      if pdt.is_categorical_dtype(vtype):
        for val in data[col].cat.categories:
          vals[val] = "{}:{}".format(col, val)
      else:
        for val in sorted(set(data[col].values)):
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
          params,
          [
            fetch_image(params, data, data.index[i])
              for i in indices
          ],
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
 * detect - detects previously used mode (the default)
 * autoencoder - learns essential features without supervision
 * predictor - learns to predict output variable(s)
 * dual - hybrid autoencoder/predictor model
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
    "--reconstruct",
    action="store_true",
    help="Recompute reconstruction error even if cached values are found."
  )
  parser.add_argument(
    "-f",
    "--features",
    action="store_true",
    help="Recompute features even if cached values are found."
  )
  parser.add_argument(
    "-F",
    "--fresh",
    action="store_true",
    help="Recompute everything. Equivalent to '-mrf'."
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
    options.reconstruct = True
    options.features = True
    # Explicitly disable all caching
    utils.toggle_caching(False)

  debug = utils.get_debug(options.quiet)

  setup_figures()

  analyze_dataset(options=vars(options))
  #utils.run_strict(analyze_dataset, options=vars(options))

  check_stattests(DEFAULT_PARAMETERS["analysis"]["confidence_baseline"])

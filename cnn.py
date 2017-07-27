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

from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

from scikits.bootstrap import ci

print("Importing keras...")
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten, Reshape
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

def NovelClustering(points, distances=None, edges=None):
  return cluster_assignments(
    points,
    condensed_multiscale(
      points,
      distances=distances,
      edges=edges,
      quiet=False
    )
  )

def safe_ci(
  data,
  statistic,
  bound=0.05,
  n_samples=10000
):
  method = "bca"
  if all(d == data[0] for d in data):
    # zero-variance case, fall back on a simple percentile interval
    method="pi"

  return ci(
    data,
    statistic,
    alpha=bound,
    n_samples=n_samples,
    method=method
  )

# Hide TensorFlow info/warnings:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Globals:

METRIC = "euclidean"

#BATCH_SIZE = 32
BATCH_SIZE = 32
PERCENT_PER_EPOCH = 1.0 # how much of the data do we feed per epoch?
#EPOCHS = 200
#EPOCHS = 50
EPOCHS = 10 # testing
#EPOCHS = 4 # fast testing
BASE_FLAT_SIZE = 512
PROBE_CUTOFF = 128 # minimum layer size to consider
#PROBE_CUTOFF = 8 # minimum layer size to consider
#CONV_SIZES = [32, 16] # network structure for convolutional layers
CONV_SIZES = [32, 16] # network structure for convolutional layers
AE_LOSS_FUNCTION = "mean_squared_error"
PR_LOSS_FUNCTION = "binary_crossentropy"
SPARSEN = True # Whether or not to regularize activity in the dense layers
REGULARIZATION_COEFFICIENT = 1e-5 # amount of l1 norm to add to the loss
#SUBTRACT_MEAN = True # whether or not to subtract means before training
INITIAL_COLORSPACE = "RGB"
USE_COLORSPACE = "HSV"
SUBTRACT_MEAN = False # whether or not to subtract means before training
ADD_CORRUPTION = False # whether or not to add corruption
NOISE_FACTOR = 0.1 # how much corruption to introduce (only if above is True)
NORMALIZE_ACTIVATION = False # Whether to add normalizing layers or not
LAPLACE_CORRECTION = False # Whether to adjust proportions by adding one
# positive and one negative point to each sample.
CONFIDENCE_LEVEL = 0.05 # Base confidence level desired

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

#IMG_DIR = os.path.join("data", "mixed", "all") # directory containing data
#IMG_DIR = os.path.join("data", "original") # directory containing data
IMG_DIR = os.path.join("data", "mii_flat") # directory containing data
#IMG_DIR = os.path.join("data", "mii_subset") # directory containing images
CSV_FILE = os.path.join("data", "csv", "miiverse_profiles.clean.csv")
INTEGER_FIELDS = [ "friends", "following", "followers", "posts", "yeahs" ]
NUMERIC_FIELDS = []
MULTI_FIELDS = { "genres": '|' }
CATEGORY_FIELDS = [ "country-code", "competence" ]
NORMALIZE_COLUMNS = [ "friends", "following", "followers", "posts", "yeahs" ]
LOGDIST_COLUMNS = [ "friends", "following", "followers", "posts", "yeahs" ]
BINARIZE_COLUMNS = {
  "friends": { -1: "private", 0: "no-friends" },
  "competence": "auto",
  "country-code": "auto",
}
FILTER_COLUMNS = [ "!private", "!no-friends" ]
#FILTER_COLUMNS = [ "!private" ]
SUBSET_SIZE = 10000
CORRELATE_WITH_ERROR = [
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
  #"no-friends",
  "genres[]",
] + [
  "genres[{}]".format(g) for g in ALL_GENRES
]
ANALYZE_PER_CLUSTER = [
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
]
SECONDARY_PER_CLUSTER = [
  "genres[{}]".format(g) for g in ALL_GENRES
]
#PREDICT_TARGET = ["private", "friends-norm"]
#PREDICT_ANALYSIS = [ "confusion", "scatter" ] 
#PREDICT_TARGET = ["friends-norm"]
#PREDICT_TARGET = ["followers-norm"]
#PREDICT_TARGET = ["friends-norm"]
#PREDICT_TARGET = ["country-code"]
#PREDICT_TARGET = ["no-friends"]
PREDICT_TARGET = ["competence"]
#PREDICT_ANALYSIS = [ "scatter" ] 
PREDICT_ANALYSIS = [ "confusion" ] 
ID_TEMPLATE = re.compile(r"([^_]+)_([^_]+)_.*") # Matches IDs in filenames
#IMAGE_SHAPE = (128, 128, 3)
IMAGE_SHAPE = (48, 48, 3)

OUTPUT_DIR = "out" # directory for output
BACKUP_NAME = "out-back-{}.zip" # output backup
NUM_BACKUPS = 4 # number of backups to keep

EXAMPLE_POOL_SIZE = 16 # how many examples per pool
LARGE_POOL_SIZE = 64
DISPLAY_ROWS = 4
FINAL_LAYER_NAME = "final_layer"
MEAN_IMAGE_FILENAME = "mean-image-{}.png"
EXAMPLE_RAW_FILENAME = "example-raw-image-{}.png"
EXAMPLE_INPUT_FILENAME = "example-input-image-{}.png"
EXAMPLE_OUTPUT_FILENAME = "example-output-image-{}.png"
BEST_FILENAME = "A-best-image-{}.png"
SAMPLED_FILENAME = "B-sampled-image-{}.png"
WORST_FILENAME = "C-worst-image-{}.png"
DUPLICATES_FILENAME = "duplicate-{}.png"
HISTOGRAM_FILENAME = "{}-histogram.pdf"
CLUSTER_SIZE_FILENAME = "cluster-sizes.pdf"
CLUSTER_STATS_FILENAME = "cluster-stats-{}.pdf"
DISTANCE_FILENAME = "{}-distances.pdf"
TSNE_FILENAME = "tsne-{}-{}v{}.pdf"
ANALYSIS_FILENAME = "analysis-{}.pdf"

TRANSFORMED_DIR = "transformed"
EXAMPLES_DIR = "examples"
DUPLICATES_DIR = "duplicates"

#CLUSTERING_METHOD = AffinityPropagation
#CLUSTERING_METHOD = DBSCAN
#CLUSTERING_METHOD = AgglomerativeClustering
CLUSTERING_METHOD = NovelClustering
CLUSTER_INPUT = "features"
#CLUSTER_INPUT = "projected"
CLUSTERS_DIR = "clusters"
CLUSTERS_REC_DIR = "clusters-rec"
MAX_CLUSTER_SAMPLES = 200 # how many clusters to visualize
SAMPLES_PER_CLUSTER = 16 # how many images from each cluster to save
CLUSTER_REP_FILENAME = "rep-{}.png"
NEIGHBORHOOD_SIZE = 4
DBSCAN_N_NEIGHBORS = 3
DBSCAN_PERCENTILE = 80
CLUSTER_SIG_SIZE = 10

ANALYZE = [
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
]

def load_data():
  items = {}
  for dp, dn, files in os.walk(IMG_DIR):
    for f in files:
      if f.endswith(".jpg") or f.endswith(".png"):
        fbase = os.path.splitext(os.path.basename(f))[0]
        match = ID_TEMPLATE.match(fbase)
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
  with open(CSV_FILE, 'r', newline='') as fin:
    reader = csv.reader(fin, dialect="excel")
    legend = next(reader)
    for i, key in enumerate(legend):
      if key in NUMERIC_FIELDS:
        values[key] = "numeric"
        types[key] = "numeric"
      elif key in INTEGER_FIELDS:
        values[key] = "integer"
        types[key] = "integer"
      elif key in MULTI_FIELDS:
        values[key] = set()
        types[key] = "multi"
      elif key in CATEGORY_FIELDS:
        values[key] = {}
        types[key] = "categorical"
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
          if col in NUMERIC_FIELDS:
            record[col] = float(val)
          elif col in INTEGER_FIELDS:
            record[col] = int(val)
          elif col in MULTI_FIELDS:
            record[col] = val
            for v in val.split(MULTI_FIELDS[col]):
              values[col].add(v)
          elif col in CATEGORY_FIELDS:
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
  for col in NORMALIZE_COLUMNS:
    add_norm_column(long_items, col)

  for col in LOGDIST_COLUMNS:
    add_log_column(long_items, col)

  # Create binary columns:
  for col in BINARIZE_COLUMNS:
    if BINARIZE_COLUMNS[col] == "auto":
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
      for val in BINARIZE_COLUMNS[col]:
        add_binary_column(long_items, col, val, BINARIZE_COLUMNS[col][val])

  precount = len(long_items["id"])

  # Filter data:
  for fil in FILTER_COLUMNS:
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

  if count > SUBSET_SIZE:
    print(
      "  Subsetting from {} to {} accepted items...".format(
        count,
        SUBSET_SIZE
      )
    )
    ilist = list(range(count))
    count = SUBSET_SIZE
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
  input_img = Input(shape=IMAGE_SHAPE)

  x = input_img

  for sz in CONV_SIZES:
    x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    if NORMALIZE_ACTIVATION:
      x = BatchNormalization()(x)

  conv_final = x
  # remember our shape, whatever it is
  # TODO: Not so hacky?
  conv_shape = conv_final._keras_shape[1:]

  x = Flatten()(x)
  flattened_size = x._keras_shape[-1]

  flat_size = BASE_FLAT_SIZE
  min_flat_size = flat_size

  # Our flat probing layers:
  while flat_size >= PROBE_CUTOFF:
    reg = None
    if SPARSEN:
      reg = l1(REGULARIZATION_COEFFICIENT)

    if flat_size // 2 < PROBE_CUTOFF: # this is the final iteration
      x = Dense(
        flat_size,
        activation='relu',
        activity_regularizer=reg,
        name=FINAL_LAYER_NAME
      )(x)
    else:
      x = Dense(
        flat_size,
        activation='relu'
        #activity_regularizer=reg
      )(x)

    if NORMALIZE_ACTIVATION:
      x = BatchNormalization()(x)

    # TODO: We welcome overfitting?
    #if flat_size // 2 < PROBE_CUTOFF: # this is the final iteration
    #  x = Dropout(0.3, name=FINAL_LAYER_NAME)(x)
    #else:
    #  x = Dropout(0.3)(x)

    # TODO: Smoother layer size reduction?
    min_flat_size = flat_size # remember last value > 1
    flat_size //= 2

  flat_final = x

  flat_size = min_flat_size * 2

  if mode in ["predictor", "dual"]:
    # In predictor mode, we narrow down to the given number of outputs
    outputs = 0
    for t in PREDICT_TARGET:
      if len(items[t].shape) > 1:
        outputs += items[t].shape[1]
      else:
        outputs += 1
    predictions = Dense(outputs, activation='relu')(x)
    if NORMALIZE_ACTIVATION:
      predictions = BatchNormalization()(predictions)

    if mode == "predictor":
      return input_img, predictions

  if mode in ["autoencoder", "dual"]:
    # In autoencoder mode, we return to the original image size:
    # TODO: construct independent return paths for each probe layer!
    while flat_size <= BASE_FLAT_SIZE:
      x = Dense(
        flat_size,
        activation='relu',
      )(x)
      # TODO: dropout on the way back up?
      flat_size *= 2

    x = Dense(flattened_size, activation='relu')(x)
    if NORMALIZE_ACTIVATION:
      x = BatchNormalization()(x)

    flat_return = x

    x = Reshape(conv_shape)(x)

    for sz in reversed(CONV_SIZES):
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    if NORMALIZE_ACTIVATION:
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
    model.compile(optimizer='adagrad', loss=AE_LOSS_FUNCTION)
  else:
    model.compile(optimizer='adagrad', loss=PR_LOSS_FUNCTION)
  return model

def get_encoding_model(auto_model):
  return Model(
    inputs=auto_model.input,
    outputs=auto_model.get_layer(FINAL_LAYER_NAME).output
  )

def load_images_into_items(items):
  # TODO: Resize images as they're loaded?
  all_images = []
  for i, filename in enumerate(items["file"]):
    utils.prbar(i / items["count"])
    img = imread(filename)
    img = img[:,:,:3] # throw away alpha channel
    convert_colorspace(img, INITIAL_COLORSPACE, USE_COLORSPACE)
    img = img / 255
    all_images.append(img)

  print() # done with progress bar

  items["image"] = np.asarray(all_images)
  items["mean_image"] = np.mean(items["image"], axis=0)
  items["image_deviation"] = items["image"] - items["mean_image"]

def create_simple_generator():
  return ImageDataGenerator().flow_from_directory(
    IMG_DIR,
    target_size=IMAGE_SHAPE[:-1],
    batch_size=1,
    shuffle=False,
    class_mode='sparse' # classes as integers
  )
  
def create_training_generator(items, mode="autoencoder"):
  src = items["image"]
  if SUBTRACT_MEAN:
    src = items["image_deviation"]
  if mode == "autoencoder":
    datagen = ImageDataGenerator() # no data augmentation (we eschew generality)

    #train_datagen = datagen.flow_from_directory(
    #  IMG_DIR,
    #  batch_size=BATCH_SIZE,
    #  class_mode='sparse' # classes as integers
    #)
    train_datagen = datagen.flow(
      src,
      src,
      batch_size=BATCH_SIZE
    )

    if ADD_CORRUPTION:
      def pairgen():
        while True:
          batch, _ = next(train_datagen)
          # Subtract mean and introduce noise to force better representations:
          for img in batch:
            corrupted = img + NOISE_FACTOR * np.random.normal(
              loc=0.0,
              scale=1.0,
              size=img.shape
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
        for t in PREDICT_TARGET:
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
      for i in range(BATCH_SIZE):
        inp, outp = next(pairgen)
        batch_in.append(inp)
        batch_out.append(outp)
      yield np.asarray(batch_in), np.asarray(batch_out)

  return batchgen(pairgen())


def train_model(model, training_gen, n):
  # Fit the model on the batches generated by datagen.flow_from_directory().
  model.fit_generator(
    training_gen,
    steps_per_epoch=int(PERCENT_PER_EPOCH * n / BATCH_SIZE),
    callbacks=[
      EarlyStopping(monitor="loss", min_delta=0, patience=0)
    ],
    epochs=EPOCHS
  )

def rate_images(items, model):
  src = items["image"]
  if SUBTRACT_MEAN:
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
    convert_colorspace(img, USE_COLORSPACE, INITIAL_COLORSPACE)
    fn = os.path.join(
      OUTPUT_DIR,
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
  path = os.path.join(OUTPUT_DIR, directory)
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
  path = os.path.join(OUTPUT_DIR, directory)
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
    os.mkdir(os.path.join(OUTPUT_DIR, TRANSFORMED_DIR), mode=0o755)
  except FileExistsError:
    pass
  ex_input, ex_output = next(generator)
  ex_raw = items["image"][:len(ex_input)]
  save_images(ex_raw, TRANSFORMED_DIR, EXAMPLE_RAW_FILENAME)
  save_images(ex_input, TRANSFORMED_DIR, EXAMPLE_INPUT_FILENAME)
  save_images(ex_output, TRANSFORMED_DIR, EXAMPLE_OUTPUT_FILENAME)
  montage_images(TRANSFORMED_DIR, EXAMPLE_RAW_FILENAME)
  montage_images(TRANSFORMED_DIR, EXAMPLE_INPUT_FILENAME)
  montage_images(TRANSFORMED_DIR, EXAMPLE_OUTPUT_FILENAME)
  collect_montages(TRANSFORMED_DIR)
  print("  ...done saving training examples...")

def analyze_duplicates(items):
  print("  Analyzing duplicates...")
  try:
    os.mkdir(os.path.join(OUTPUT_DIR, DUPLICATES_DIR), mode=0o755)
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

  representatives = representatives[-LARGE_POOL_SIZE:]
  duplications = duplications[-LARGE_POOL_SIZE:]

  save_images(
    representatives,
    DUPLICATES_DIR,
    DUPLICATES_FILENAME,
    labels=duplications
  )
  montage_images(DUPLICATES_DIR, DUPLICATES_FILENAME)

  plt.clf()
  n, bins, patches = plt.hist(items["duplicates"], 100)
  #plt.plot(bins)
  plt.xlabel("Number of Duplicates")
  plt.ylabel("Number of Images")
  plt.savefig(
    os.path.join(OUTPUT_DIR, HISTOGRAM_FILENAME.format("duplicates"))
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

  if "duplicates" in ANALYZE:
    print('-'*80)
    analyze_duplicates(items)

  # We want only nearest-neighbors for novel clustering (otherwise sorting
  # edges is too expensive).
  if method == NovelClustering:
    results["nearest_neighbors"] = np.argsort(
      items["distances"],
      axis=1
    )[:,1:NEIGHBORHOOD_SIZE+1]
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
    )[:,1:DBSCAN_N_NEIGHBORS+1]
    results["outer_distances"] = results["ordered_distances"][
      :,
      DBSCAN_N_NEIGHBORS-1
    ]
    results["outer_distances"] = np.sort(results["outer_distances"])
    smp = results["outer_distances"][::items["count"]//10]
    print("   Distance sample:")
    print(smp)
    print("  ...done.")
    #closest, min_dist = pairwise.pairwise_distances_argmin_min(
    #  items[CLUSTER_INPUT],
    #  items[CLUSTER_INPUT],
    #  metric=metric
    #)
    clustering_distance = 0
    perc = DBSCAN_PERCENTILE
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
        DBSCAN_N_NEIGHBORS,
        clustering_distance
      )
    )

  if method == DBSCAN:
    model = DBSCAN(
      eps=clustering_distance,
      min_samples=DBSCAN_N_NEIGHBORS,
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
      items[CLUSTER_INPUT],
      edges=edges
    )
  else:
    fit = model.fit(items[CLUSTER_INPUT])
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

def main(options):
  # Seed random number generator (hopefully improve image loading times via
  # disk cache?)
  print("Random seed is: {}".format(options.seed))
  random.seed(options.seed)
  # Backup old output directory and create a new one:
  print("Managing output backups...")
  bn = BACKUP_NAME.format(NUM_BACKUPS - 1)
  if os.path.exists(bn):
    print("Removing oldest backup '{}' (keeping {}).".format(bn, NUM_BACKUPS))
    os.remove(bn)
  for i in range(NUM_BACKUPS)[-2::-1]:
    bn = BACKUP_NAME.format(i)
    nbn = BACKUP_NAME.format(i+1)
    if os.path.exists(bn):
      print("  ...found.")
      os.rename(bn, nbn)

  if os.path.exists(OUTPUT_DIR):
    bn = BACKUP_NAME.format(0)
    shutil.make_archive(bn[:-4], 'zip', OUTPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)

  try:
    os.mkdir(OUTPUT_DIR, mode=0o755)
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
  if "mean_image" in ANALYZE:
    print("  Saving mean image...")
    save_images([items["mean_image"]], ".", MEAN_IMAGE_FILENAME)
  print("  ...done loading images.")
  print('-'*80)

  if options.mode == "detect":
    options.mode = utils.cached_value(
      lambda: "autoencoder",
      "mode",
      "str",
      debug=print
    )
  else:
    print("Selected mode '{}'".format(options.mode))
    utils.store_cache(options.mode, "mode", "str")

  if options.mode == "dual":
    required_models = [
      "autoencoder-model",
      "predictor-model"
    ]
  else:
    required_models = [ options.mode + "-model" ]

  print('-'*80)
  print("Acquiring {} model...".format(options.mode))
  if options.mode == "dual":
    def get_models():
      global ANALYZE
      nonlocal items, options
      print("  Creating models...")
      inp, comp = setup_computation(items, mode="dual")
      ae_model = compile_model(inp, comp[0], mode="autoencoder")
      pr_model = compile_model(inp, comp[1], mode="predictor")
      print("  ...done creating models.")
      print("  Creating training generators...")
      ae_train_gen = create_training_generator(items, mode="autoencoder")
      pr_train_gen = create_training_generator(items, mode="predictor")
      print("  ...done creating training generators.")
      if "training_examples" in ANALYZE:
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
      override=options.model,
      debug=print
    )
  else:
    def get_model():
      global ANALYZE
      nonlocal items, options
      print("  Creating model...")
      inp, comp = setup_computation(items, mode=options.mode)
      model = compile_model(inp, comp, mode=options.mode)
      print("  ...done creating model.")
      print("  Creating training generator...")
      train_gen = create_training_generator(items, mode=options.mode)
      print("  ...done creating training generator.")
      if "training_examples" in ANALYZE:
        save_training_examples(items, train_gen)
      print("  Training model...")
      if options.mode == "dual":
        train_model(ae_model, ae_train_gen, items["count"])
        train_model(pr_model, pr_train_gen, items["count"])
      else:
        train_model(model, train_gen, items["count"])
      print("  ...done training model.")
      return model

    model = utils.cached_value(
      get_model,
      options.mode + "-model",
      typ="h5",
      override=options.model,
      debug=print
    )

  print('-'*80)
  if options.mode == "dual":
    print("Got models:")
    print(ae_model.summary())
    print("\n...and...\n")
    print(pr_model.summary())
  else:
    print("Got model:")
    print(model.summary())
  print('-'*80)

  if options.mode == "dual":
    # DEBUG:
    test_autoencoder(items, ae_model, options)
    test_predictor(items, pr_model, options)
  elif options.mode == "autoencoder":
    test_autoencoder(items, model, options)
  elif options.mode == "predictor":
    test_predictor(items, model, options)
  else:
    print(
      "Error: Unknown mode '{}'. No tests to run.".format(options.mode),
      file=sys.stderr
    )

def analyze_correlations(items, columns, against):
  p_threshold = 1 / (20 * len(columns))
  for col in columns:
    other = items[col]

    r, p = pearsonr(items[against], other)
    if p < p_threshold:
      print("  '{}': {}  (p={})".format(col, r, p))
      plt.figure()
      plt.scatter(items[against], other, s=0.25)
      plt.xlabel(against)
      plt.ylabel(col)
    else:
      print("    '{}': not significant (p={})".format(col, p))

  plt.show()

def relevant_statistic(items, col):
  vtype = items["types"][col]

  def simple_proportion(data):
    return np.sum(data) / len(data)

  def laplace_proportion(data):
    return (np.sum(data) + 1) / (len(data) + 2)

  if vtype == "boolean":
    if LAPLACE_CORRECTION:
      stat = laplace_proportion
    else:
      stat = simple_proportion
  elif vtype == "categorical":
    if len(items["values"][col]) == 2:
      if LAPLACE_CORRECTION:
        stat = laplace_proportion
      else:
        stat = simple_proportion
    else:
      stat = np.average
      raise RuntimeWarning(
"Warning: Using 'average' stat for multi-category column '{}'."
        .format(col, vtype)
      )
  elif vtype in ("integer", "numeric"):
    stat = np.average
  else:
    stat = np.average
    raise RuntimeWarning(
"Warning: Using 'average' stat for column '{}' with type '{}'."
      .format(col, vtype)
    )
  return stat

def analyze_cluster_stats(items, which_stats, secondary_stats):
  cstats = {
    c: {} for c in items["cluster_ids"]
  }

  for c in cstats:
    indices = items["cluster"] == c
    for col in which_stats:
      cstats[c][col] = items[col][indices]
    for col in secondary_stats:
      cstats[c][col] = items[col][indices]
    cstats[c]["size"] = len(cstats[c][which_stats[0]])

  plt.close()
  big_enough = [
    c
      for c in cstats
      if (
        cstats[c]["size"] >= CLUSTER_SIG_SIZE
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
      CLUSTER_SIG_SIZE
    )
  )

  # Compute desired confidence bounds:
  ntests = len(which_stats) + len(secondary_stats)
  total_tests = nbig * ntests

  # The Šidák correction:
  # TODO: Bonferroni correction? That requires a whole other algorithm though.
  shared_alpha = 1 - (1 - CONFIDENCE_LEVEL)**(1 / total_tests)
  bootstrap_samples = int(2/shared_alpha)
  print(
    (
"  ...testing {} properties of {} clusters ({} total tests)...\n"
"  ...setting individual α={} for joint α={}...\n"
"  ...using {} samples for bootstrapping (2/α)..."
    ).format(
      ntests, nbig, total_tests,
      shared_alpha, CONFIDENCE_LEVEL,
      bootstrap_samples
    )
  )

  print("  ...extracting cluster stats...")
  for i, c in enumerate(big_enough):
    utils.prbar(i / len(big_enough))
    #for col in which_stats + list(secondary_stats):
    for col in which_stats + secondary_stats:
      if col not in cinfo:
        cinfo[col] = []

      x = cstats[c][col]

      stat = relevant_statistic(items, col)

      cstats[c][col + "_stat"] = stat(x)
      cstats[c][col + "_ci"] = safe_ci(
        x,
        stat,
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
  test = which_stats + secondary_stats

  small = { col: set() for col in test }
  large = { col: set() for col in test }
  diff = { col: set() for col in test }
  stat = relevant_statistic(items, col)
  overall_stats = {
    col: relevant_statistic(items, col)(items[col]) for col in test
  }
  print("  ...bootstrapping overall means...")
  overall_cis = {}
  for i, col in enumerate(test):
    utils.prbar(i / len(test))
    overall_cis[col] = safe_ci(
      items[col],
      relevant_statistic(items, col),
      bound=shared_alpha,
      n_samples=bootstrap_samples
    )
  print()
  print("  ...done bootstrapping overall means.")

  print("  ...computing cluster property significance...")
  for i, col in enumerate(test):
    utils.prbar(i / len(test))
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
        # TODO: This normalization is *REALLY* sketchy and not supported by any
        # literature whatsoever! Find another way of comparing things.
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
        # TODO: Something else for Poisson-distributed variables?
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

    if col in test and any((small[col], large[col], diff[col])):
      print("Outliers for '{}':".format(col))
      print("  small:", sorted(list(small[col])))
      print("  large:", sorted(list(large[col])))
      print("  diff:", sorted(list(diff[col])))

    cinfo[col] = np.asarray(cinfo[col])
    # x values are just integers:
    x = np.arange(nbig)
    cids = cinfo[col][:,0]
    stats = cinfo[col][:,1]
    bot = cinfo[col][:,2]
    top = cinfo[col][:,3]
    if col in test:
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
      os.path.join(OUTPUT_DIR, CLUSTER_STATS_FILENAME.format(col))
    )
  print("  ...done plotting & summarizing.")

  montage_images(".", CLUSTER_STATS_FILENAME)

def reconstruct_image(items, img, model):
  img = img.reshape((1,) + img.shape) # pretend it's a batch
  pred = model.predict(img)[0]
  if SUBTRACT_MEAN:
    pred += items["mean_image"]
  return pred

def test_autoencoder(items, model, options):
  items["rating"] = utils.cached_value(
    lambda: rate_images(items, model),
    "ratings",
    override=options.rank,
    debug=print
  )
  print('-'*80)
  items["norm_rating"] = items["rating"] / np.max(items["rating"])

  # Save the best images and their reconstructions:
  if "reconstructions" in ANALYZE:
    print("Saving example images...")
    try:
      os.mkdir(os.path.join(OUTPUT_DIR, EXAMPLES_DIR), mode=0o755)
    except FileExistsError:
      pass
    sorted_images = images_sorted_by_accuracy(items)

    best = sorted_images[:EXAMPLE_POOL_SIZE]
    worst = sorted_images[-EXAMPLE_POOL_SIZE:]
    rnd = items["image"][:]
    random.shuffle(rnd)
    rnd = rnd[:EXAMPLE_POOL_SIZE]

    for iset, fnt in zip(
      [best, worst, rnd],
      [BEST_FILENAME, WORST_FILENAME, SAMPLED_FILENAME]
    ):
      save_images(iset, EXAMPLES_DIR, fnt)
      rec_images = []
      for img in iset:
        rec_images.append(reconstruct_image(items, img, model))
      save_images(rec_images, EXAMPLES_DIR, "rec-" + fnt)
      montage_images(EXAMPLES_DIR, fnt)
      montage_images(EXAMPLES_DIR, "rec-" + fnt)
    collect_montages(EXAMPLES_DIR)
    print("  ...done.")

  print('-'*80)
  src = items["image"]
  if SUBTRACT_MEAN:
    src = items["image_deviation"]
  items["features"] = utils.cached_value(
    lambda: get_features(src, model),
    "features",
    override=options.features,
    debug=print
  )

  print('-'*80)
  tsne = TSNE(n_components=2, random_state=0)
  items["projected"] = utils.cached_value(
    lambda: tsne.fit_transform(items["features"]),
    "projected",
    override=options.project,
    debug=print
  )


  print('-'*80)
  if CLUSTERING_METHOD == DBSCAN:
    (
      items["cluster"],
      items["cluster_ids"],
      items["cluster_sizes"],
      items["ordered_distances"],
      items["outer_distances"],
      items["core_mask"],
    ) = utils.cached_values(
      lambda: get_clusters(CLUSTERING_METHOD, items, CLUSTER_INPUT, METRIC),
      (
        "clusters",
        "cluster_ids",
        "cluster_sizes",
        "ordered_distances",
        "outer_distances",
        "core_mask",
      ),
      ("pkl", "pkl", "pkl", "pkl", "pkl", "pkl"),
      override=options.cluster,
      debug=print
    )
  else:
    (
      items["cluster"],
      items["cluster_ids"],
      items["cluster_sizes"],
    ) = utils.cached_values(
      lambda: get_clusters(CLUSTERING_METHOD, items, CLUSTER_INPUT),
      ("clusters", "cluster_ids", "cluster_sizes"),
      ("pkl", "pkl", "pkl"),
      override=options.cluster,
      debug=print
    )

  # Plot a histogram of error values for all images:
  if "reconstruction_error" in ANALYZE:
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
    plt.savefig(os.path.join(OUTPUT_DIR, HISTOGRAM_FILENAME.format("error")))
    plt.clf()
    print("  ...done.")

  if "reconstruction_correlations" in ANALYZE:
    print('-'*80)
    print("Computing reconstruction correlations...")
    analyze_correlations(items, CORRELATE_WITH_ERROR, "norm_rating")
    print("  ...done.")

  if "typicality_correlations" in ANALYZE:
    print('-'*80)
    print("Computing typicality...")
    items["typicality"] = utils.cached_value(
      lambda: typicality(
        items["features"],
        quiet=False,
        metric=METRIC
      ),
      "typicality",
      "pkl",
      debug=print
    )
    analyze_correlations(items, CORRELATE_WITH_ERROR, "typicality")
    print("  ...done.")

  # Plot the t-SNE results:
  if "tSNE" in ANALYZE:
    print('-'*80)
    print("Plotting t-SNE results...")
    cycle = palettable.tableau.Tableau_20.mpl_colors
    colors = []
    alt_colors = []
    first_target = PREDICT_TARGET[0]
    vtype = items["types"][first_target]
    if vtype in ["numeric", "integer"]:
      tv = items[first_target]
      norm = (tv - np.min(tv)) / (np.max(tv) - np.min(tv))
      cmap = plt.get_cmap("plasma")
      #cmap = plt.get_cmap("viridis")
      for v in norm:
        colors.append(cmap(v))
    elif vtype == "boolean":
      for v in items[first_target]:
        colors.append(cycle[v % len(cycle)])
    else: # including categorical items
      mapping = {}
      max_so_far = 0
      for v in items[first_target]:
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
    if CLUSTERING_METHOD == DBSCAN:
      sizes = items["core_mask"]*0.75 + 0.25

    for cl in items["cluster"]:
      if cl == -1:
        alt_colors.append((0.0, 0.0, 0.0)) # black
      else:
        alt_colors.append(cycle[cl % len(cycle)])

    if "typicality" in items:
      typ_colors = []
      cmap = plt.get_cmap("plasma")
      for t in items["typicality"]:
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
        items["projected"][:,x],
        items["projected"][:,y],
        s=0.25,
        c=colors
      )
      plt.xlabel("t-SNE {}".format("xyz"[x]))
      plt.ylabel("t-SNE {}".format("xyz"[y]))
      plt.savefig(os.path.join(OUTPUT_DIR, TSNE_FILENAME.format("true", x, y)))

      # Plot using guessed colors:
      plt.clf()
      ax = plt.scatter(
        items["projected"][:,x],
        items["projected"][:,y],
        s=sizes,
        c=alt_colors
      )
      plt.xlabel("t-SNE {}".format("xyz"[x]))
      plt.ylabel("t-SNE {}".format("xyz"[y]))
      plt.savefig(
        os.path.join(OUTPUT_DIR, TSNE_FILENAME.format("learned", x, y))
      )

      # Plot typicality:
      if "typicality" in items:
        plt.clf()
        ax = plt.scatter(
          items["projected"][:,x],
          items["projected"][:,y],
          s=0.25,
          c=typ_colors
        )
        plt.xlabel("t-SNE {}".format("xyz"[x]))
        plt.ylabel("t-SNE {}".format("xyz"[y]))
        plt.savefig(os.path.join(OUTPUT_DIR, TSNE_FILENAME.format("typ", x, y)))

    # TODO: Less hacky here
    montage_images(".", TSNE_FILENAME.format("*", "*", "{}"))
    print("  ...done.")

  # Plot a histogram of pairwise distance values (if we computed them):
  if "distance_histograms" in ANALYZE and CLUSTERING_METHOD == DBSCAN:
    print('-'*80)
    print(
      "Plotting distance histograms...".format(DBSCAN_N_NEIGHBORS)
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
          OUTPUT_DIR,
          HISTOGRAM_FILENAME.format("distance-{}".format(col))
        )
      )
      plt.clf()
    montage_images(".", HISTOGRAM_FILENAME.format("distance-{}"))
    print("  ...done.")

  if "distances" in ANALYZE:
    print('-'*80)
    print("Plotting distances...")
    plt.plot(items["outer_distances"])
    plt.xlabel("Index")
    plt.ylabel("Distance to {} Neighbor".format(ordinal(DBSCAN_N_NEIGHBORS)))
    plt.savefig(
      os.path.join(
        OUTPUT_DIR,
        DISTANCE_FILENAME.format(ordinal(DBSCAN_N_NEIGHBORS))
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
          OUTPUT_DIR,
          HISTOGRAM_FILENAME.format("distance-ratio")
        )
      )
      plt.clf()
    else:
      raise RuntimeWarning(
        "Warning: no distance ratio information available (too many clones)."
      )
    print("  ...done.")

  if "cluster_sizes" in ANALYZE:
    print('-'*80)
    # Plot cluster sizes
    print("Plotting cluster sizes...")
    just_counts = list(reversed(sorted(list(items["cluster_sizes"].values()))))
    plt.clf()
    plt.scatter(range(len(just_counts)), just_counts, s=2.5, c="k")
    plt.xlabel("cluster")
    plt.ylabel("cluster size")
    plt.savefig(os.path.join(OUTPUT_DIR, CLUSTER_SIZE_FILENAME))
    plt.clf()
    print("  ...done.")

  if "cluster_statistics" in ANALYZE:
    print('-'*80)
    # Summarize statistics per-cluster:
    print("Summarizing clustered statistics...")
    analyze_cluster_stats(items, ANALYZE_PER_CLUSTER, SECONDARY_PER_CLUSTER)
    print("  ...done.")

  if "cluster_samples" in ANALYZE:
    print('-'*80)
    # Show some of the clustering results (TODO: better):
    print("Sampling clustered images...")
    try:
      os.mkdir(os.path.join(OUTPUT_DIR, CLUSTERS_DIR), mode=0o755)
    except FileExistsError:
      pass
    try:
      os.mkdir(os.path.join(OUTPUT_DIR, CLUSTERS_REC_DIR), mode=0o755)
    except FileExistsError:
      pass

    # TODO: Get more representative images?
    nvc = min(MAX_CLUSTER_SAMPLES, len(items["cluster_ids"]))
    shuf = list(zip(items["image"], items["cluster"]))
    random.shuffle(shuf)

    for i, c in enumerate(list(items["cluster_ids"])[:MAX_CLUSTER_SAMPLES]):
      utils.prbar(i / nvc)
      cluster_images = []
      rec_images = []
      for i, (img, cluster) in enumerate(shuf):
        if cluster == c:
          rec = reconstruct_image(items, img, model)
          cluster_images.append(img)
          rec_images.append(rec)
          if len(cluster_images) >= SAMPLES_PER_CLUSTER:
            break
      if c == -1:
        thisdir = os.path.join(
          CLUSTERS_DIR,
          "outliers-({})".format(items["cluster_sizes"][c] + 1)
        )
        recdir = os.path.join(
          CLUSTERS_REC_DIR,
          "outliers-({})".format(items["cluster_sizes"][c] + 1)
        )
      else:
        thisdir = os.path.join(
          CLUSTERS_DIR,
          "cluster-{}-({})".format(c, items["cluster_sizes"][c] + 1)
        )
        recdir = os.path.join(
          CLUSTERS_REC_DIR,
          "cluster-{}-({})".format(c, items["cluster_sizes"][c] + 1)
        )
      try:
        os.mkdir(os.path.join(OUTPUT_DIR, thisdir), mode=0o755)
      except FileExistsError:
        pass
      try:
        os.mkdir(os.path.join(OUTPUT_DIR, recdir), mode=0o755)
      except FileExistsError:
        pass
      save_images(cluster_images, thisdir, CLUSTER_REP_FILENAME)
      save_images(rec_images, recdir, CLUSTER_REP_FILENAME)
      montage_images(
        thisdir,
        CLUSTER_REP_FILENAME,
        label=items["cluster_sizes"][c]
      )
      montage_images(
        recdir,
        CLUSTER_REP_FILENAME,
        label=items["cluster_sizes"][c]
      )

    print() # done with the progress bar
    print("  ...creating combined cluster sample image...")
    collect_montages(CLUSTERS_DIR, label_dirnames=True)
    collect_montages(CLUSTERS_REC_DIR, label_dirnames=True)
    print("  ...done.")

def test_predictor(items, model, options):
  if "prediction_accuracy" in ANALYZE:
    print('-'*80)
    print("Analyzing prediction accuracy...")
    print("  ...there are {} samples...".format(items["count"]))
    true = np.stack([items[t] for t in PREDICT_TARGET])

    src = items["image"]
    if SUBTRACT_MEAN:
      src = items["image_deviation"]

    rpred = model.predict(src)
    predicted = np.asarray(rpred.reshape(true.shape), dtype=float)

    for i, t in enumerate(PREDICT_ANALYSIS):
      target = PREDICT_TARGET[i]
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
        plt.savefig(os.path.join(OUTPUT_DIR, ANALYSIS_FILENAME.format(target)))
      elif t == "scatter":
        plt.clf()
        plt.scatter(x, y, s=0.25)
        fit = np.polyfit(x, y, deg=1)
        plt.plot(x, fit[0]*x + fit[1], color="red", linewidth=0.1)
        plt.xlabel("True {}".format(target.title()))
        plt.ylabel("Predicted {}".format(target.title()))
        plt.savefig(os.path.join(OUTPUT_DIR, ANALYSIS_FILENAME.format(target)))

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

  main(options)
  #utils.run_strict(main, options)

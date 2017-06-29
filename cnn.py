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

import math
import os
import glob
import sys
import subprocess
import shutil
import optparse
import pickle
import random

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1

import numpy as np
import scipy.misc

import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise

import palettable

from cluster import cluster as NovelClustering

# Globals:

#BATCH_SIZE = 32
BATCH_SIZE = 32
PERCENT_PER_EPOCH = 1.0 # how much of the data do we feed per epoch?
#EPOCHS = 200
#EPOCHS = 50
#EPOCHS = 10 # testing
EPOCHS = 4 # fast testing
BASE_FLAT_SIZE = 512
PROBE_CUTOFF = 128 # minimum layer size to consider
#PROBE_CUTOFF = 8 # minimum layer size to consider
CONV_SIZES = [32, 16] # network structure for convolutional layers
SPARSEN = True # Whether or not to regularize activity in the dense layers
REGULARIZATION_COEFFICIENT = 1e-5 # amount of l1 norm to add to the loss
SUBTRACT_MEAN = False # whether or not to subtract means before training
ADD_CORRUPTION = False # whether or not to add corruption
NOISE_FACTOR = 0.1 # how much corruption to introduce (only if above is True)
NORMALIZE_ACTIVATION = False # Whether to add normalizing layers or not

#DATA_DIR = os.path.join("data", "mixed", "all") # directory containing data
#DATA_DIR = os.path.join("data", "original") # directory containing data
#DATA_DIR = os.path.join("data", "mii_data") # directory containing data
DATA_DIR = os.path.join("data", "mii_subset_flat") # directory containing data
#IMAGE_SHAPE = (128, 128, 3)
IMAGE_SHAPE = (48, 48, 3)

OUTPUT_DIR = "out" # directory for output
BACKUP_NAME = "out-back-{}.zip" # output backup
NUM_BACKUPS = 4 # number of backups to keep

EXAMPLE_POOL_SIZE = 16 # how many examples per pool
DISPLAY_ROWS = 4
CACHE_DIR = "cache" # directory for cache files
MODEL_CACHE = os.path.join(CACHE_DIR, "cached-model.h5")
RATINGS_CACHE = os.path.join(CACHE_DIR, "cached-ratings.pkl")
FEATURES_CACHE = os.path.join(CACHE_DIR, "cached-features.pkl")
PROJECTION_CACHE = os.path.join(CACHE_DIR, "cached-projection.pkl")
CLUSTER_CACHE = os.path.join(CACHE_DIR, "cached-clusters.pkl")
FINAL_LAYER_NAME = "final_layer"
MEAN_IMAGE_FILENAME = "mean-image-{}.png"
EXAMPLE_IMAGE_FILENAME = "example-whitened-image-{}.png"
BEST_FILENAME = "A-best-image-{}.png"
SAMPLED_FILENAME = "B-sampled-image-{}.png"
WORST_FILENAME = "C-worst-image-{}.png"
HISTOGRAM_FILENAME = "{}-histogram.pdf"
DISTANCE_FILENAME = "{}-distances.pdf"
TSNE_FILENAME = "tsne-{}-{}v{}.pdf"

TRANSFORMED_DIR = "transformed"
EXAMPLES_DIR = "examples"

#CLUSTERING_METHOD = AffinityPropagation
#CLUSTERING_METHOD = DBSCAN
#CLUSTERING_METHOD = AgglomerativeClustering
CLUSTERING_METHOD = NovelClustering
CLUSTER_INPUT = "features"
#CLUSTER_INPUT = "projected"
CLUSTERS_DIR = "clusters"
MAX_CLUSTER_SAMPLES = 200 # how many clusters to visualize
SAMPLES_PER_CLUSTER = 16 # how many images from each cluster to save
CLUSTER_REP_FILENAME = "rep-{}.png"
DBSCAN_N_NEIGHBORS = 3
DBSCAN_PERCENTILE = 25

N_EXAMPLES = 0
for dp, dn, files in os.walk(DATA_DIR):
  for f in files:
    if f.endswith(".jpg") or f.endswith(".png"):
      N_EXAMPLES += 1

def ordinal(n):
  if 11 <= n <= 19:
    return str(n) + "th"
  s = str(n)
  last = int(s[-1])
  if 1 <= last <= 3:
    return s + ("st", "nd", "rd")[last-1]
  return s + "th"

def setup_computation():
  # TODO: How does resizing things affect them?
  input_img = Input(shape=IMAGE_SHAPE)

  x = input_img
  print("input:", x._keras_shape)

  for sz in CONV_SIZES:
    x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    if NORMALIZE_ACTIVATION:
      x = BatchNormalization()(x)
    print("sz", sz, ":", x._keras_shape)

  conv_final = x
  # remember our shape, whatever it is
  # TODO: Not so hacky?
  print("conv_final:", conv_final._keras_shape)
  conv_shape = conv_final._keras_shape[1:]

  x = Flatten()(x)
  flattened_size = x._keras_shape[-1]
  print("flat_top:", flattened_size)

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
    print("flat", flat_size, ":", x._keras_shape)

    # TODO: Smoother layer size reduction?
    min_flat_size = flat_size # remember last value > 1
    flat_size //= 2

  flat_final = x

  flat_size = min_flat_size * 2

  # TODO: construct independent return paths for each probe layer!
  while flat_size <= BASE_FLAT_SIZE:
    x = Dense(
      flat_size,
      activation='relu',
    )(x)
    print("flat_up", flat_size, ":", x._keras_shape)
    # TODO: dropout on the way back up?
    flat_size *= 2

  x = Dense(flattened_size, activation='relu')(x)
  if NORMALIZE_ACTIVATION:
    x = BatchNormalization()(x)
  print("flat_return:", x._keras_shape)

  flat_return = x

  x = Reshape(conv_shape)(x)
  print("reshaped:", x._keras_shape)

  for sz in reversed(CONV_SIZES):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(sz, (3, 3), activation='relu', padding='same')(x)
    print("sz_rev", sz, ":", x._keras_shape)

  x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
  if NORMALIZE_ACTIVATION:
    x = BatchNormalization()(x)
  print("decoded:", x._keras_shape)

  decoded = x

  return input_img, decoded

def compile_model(input, autoencoded):
  model = Model(input, autoencoded)
  # TODO: These choices?
  #model.compile(optimizer='adadelta', loss='mean_squared_error')
  model.compile(optimizer='adagrad', loss='mean_squared_error')
  return model

def get_encoding_model(auto_model):
  return Model(
    inputs=auto_model.input,
    outputs=auto_model.get_layer(FINAL_LAYER_NAME).output
  )

def create_training_generator(raw_images, mean_image):
  datagen = ImageDataGenerator( # no data augmentation (we eschew generality)
    rescale=1/255 # normalize RGB values to [0,1]
  )

  train_datagen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SHAPE[:-1], # should be a power of two
    batch_size=BATCH_SIZE,
    class_mode='sparse' # classes as integers
  )

  def pairgen():
    while True:
      batch, _ = next(train_datagen)
      # Subtract mean and introduce noise to force better representations:
      for img in batch:
        if SUBTRACT_MEAN:
          norm = img - mean_image
        else:
          norm = img
        if ADD_CORRUPTION:
          corrupted = norm + NOISE_FACTOR * np.random.normal(
            loc=0.0,
            scale=1.0,
            size=img.shape
          )
          yield (corrupted, norm)
        else:
          yield (norm, norm)
  
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

def create_simple_generator():
  return ImageDataGenerator(
    rescale=1/255 # normalize RGB values to [0,1]
  ).flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SHAPE[:-1],
    batch_size=1,
    shuffle=False,
    class_mode='sparse' # classes as integers
  )

def train_model(model, training_gen):
  # Fit the model on the batches generated by datagen.flow_from_directory().
  model.fit_generator(
    training_gen,
    steps_per_epoch=int(PERCENT_PER_EPOCH * N_EXAMPLES / BATCH_SIZE),
    callbacks=[
      EarlyStopping(monitor="loss", min_delta=0, patience=0)
    ],
    epochs=EPOCHS
  )

def prbar(progress):
  pbwidth = 70
  sofar = int(pbwidth * progress)
  left = pbwidth - sofar - 1
  print("\r[" + "="*sofar + ">" + "-"*left + "]", end="")

def rate_images(model, images, mean_image):
  ratings = []
  print("There are {} example images.".format(N_EXAMPLES))
  progress = 0
  for i, img in enumerate(images):
    prbar(i / N_EXAMPLES)
    if SUBTRACT_MEAN:
      norm = img - mean_image
    else:
      norm = img
    norm = norm.reshape((1,) + norm.shape) # pretend it's a batch
    ratings.append(model.test_on_batch(norm, norm))

  print() # done with the progress bar
  return ratings

def get_images(simple_gen):
  images = []
  classes = []
  print("There are {} example images.".format(N_EXAMPLES))

  for i in range(N_EXAMPLES):
    img, cls = next(simple_gen)
    images.append(img[0])
    classes.append(cls[0])

  return images, classes

def sorted_by_accuracy(images, ratings):
  return [
    pair[0] for pair in
      sorted(list(zip(images, ratings)), key=lambda pair: pair[1])
  ]

def save_images(images, directory, name_template):
  for i in range(len(images)):
    img = scipy.misc.toimage(images[i], cmin=0.0, cmax=1.0)
    img.save(os.path.join(OUTPUT_DIR, directory, name_template.format(i)))

def montage_images(directory, name_template):
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

def collect_montages(directory):
  path = os.path.join(OUTPUT_DIR, directory)
  montages = []
  for root, dirs, files in os.walk(path):
    for f in files:
      if "montage" in f:
        montages.append(os.path.relpath(os.path.join(root, f)))
  montages.sort()
  subprocess.run([
    "montage",
      "-geometry",
      "+4+4",
  ] + montages + [
      "{}/combined-montage.png".format(path)
  ])

def get_features(images, model):
  encoder = get_encoding_model(model)
  return encoder.predict(np.asarray(images))

def count_clusters(clusters):
  valid_clusters = set(clusters)
  cluster_counts = {}
  for v in valid_clusters:
    for c in clusters:
      if c == v:
        if v in cluster_counts:
          cluster_counts[v] += 1
        else:
          cluster_counts[v] = 1

  return cluster_counts

def main(options):
  # Backup old output directory and create a new one:
  print("Managing output backups...")
  if options.pause:
    input("  Ready to continue (press enter) > ")
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


  print('-'*80)
  print("Loading images...")
  if options.pause:
    input("  Ready to continue (press enter) > ")
  simple_gen = create_simple_generator()
  images, classes = get_images(simple_gen)
  print("Computing mean image for standardization.")
  mean_image = np.mean(images, axis=0)
  save_images([mean_image], ".", MEAN_IMAGE_FILENAME)
  print("  ...done loading images.")
  print('-'*80)

  if not os.path.exists(MODEL_CACHE) or options.model:
    print('-'*80)
    print("Generating fresh model...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    inp, auto = setup_computation()
    autoencoder = compile_model(inp, auto)
    print("  Fitting generator to normalize images...")
    train_gen = create_training_generator(images, mean_image)
    print("  ...done creating normalizing generator.")
    try:
      os.mkdir(os.path.join(OUTPUT_DIR, TRANSFORMED_DIR), mode=0o755)
    except FileExistsError:
      pass
    ex_batch, _ = next(train_gen)
    save_images(ex_batch, TRANSFORMED_DIR, EXAMPLE_IMAGE_FILENAME)
    montage_images(TRANSFORMED_DIR, EXAMPLE_IMAGE_FILENAME)
    print("  Training model...")
    train_model(autoencoder, train_gen)
    autoencoder.save(MODEL_CACHE)
    print("  ...done training model.")
    print('-'*80)
  else:
    print('-'*80)
    print("Using stored model...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    autoencoder = keras.models.load_model(MODEL_CACHE)
    print("  ...done loading model.")
    print('-'*80)

  print(autoencoder.summary())

  if not os.path.exists(RATINGS_CACHE) or options.rank:
    print('-'*80)
    print("Rating all images...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    ratings = rate_images(autoencoder, images, mean_image)
    with open(RATINGS_CACHE, 'wb') as fout:
      pickle.dump(ratings, fout)
    print("  ...done rating images.")
    print('-'*80)
  else:
    print('-'*80)
    print("Loading images and cached ratings...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    with open(RATINGS_CACHE, 'rb') as fin:
      ratings = pickle.load(fin)
    print("  ...done loading images and ratings.")
    print('-'*80)

  # Save the best images and their reconstructions:
  print("Saving example images...")
  if options.pause:
    input("  Ready to continue (press enter) > ")
  try:
    os.mkdir(os.path.join(OUTPUT_DIR, EXAMPLES_DIR), mode=0o755)
  except FileExistsError:
    pass
  sorted_images = sorted_by_accuracy(images, ratings)

  best = sorted_images[:EXAMPLE_POOL_SIZE]
  worst = sorted_images[-EXAMPLE_POOL_SIZE:]
  rnd = images[:]
  random.shuffle(rnd)
  rnd = rnd[:EXAMPLE_POOL_SIZE]

  for iset, fnt in zip(
    [best, worst, rnd],
    [BEST_FILENAME, WORST_FILENAME, SAMPLED_FILENAME]
  ):
    save_images(iset, EXAMPLES_DIR, fnt)
    rec_images = []
    for img in iset:
      if SUBTRACT_MEAN:
        norm = img - mean_image
      else:
        norm = img
      norm = norm.reshape((1,) + norm.shape) # pretend it's a batch
      pred = autoencoder.predict(norm)[0]
      if SUBTRACT_MEAN:
        pred += mean_image
      rec_images.append(pred)
    save_images(rec_images, EXAMPLES_DIR, "rec-" + fnt)
    montage_images(EXAMPLES_DIR, fnt)
    montage_images(EXAMPLES_DIR, "rec-" + fnt)
  collect_montages(EXAMPLES_DIR)
  print("  ...done.")

  if not os.path.exists(FEATURES_CACHE) or options.features:
    print('-'*80)
    print("Computing image features...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    features = get_features(images, autoencoder)
    with open(FEATURES_CACHE, 'wb') as fout:
      pickle.dump(features, fout)
    print("  ...done computing image features.")
    print('-'*80)
  else:
    print('-'*80)
    print("Loading cached image features...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    with open(FEATURES_CACHE, 'rb') as fin:
      features = pickle.load(fin)
    print("  ...done loading image features.")
    print('-'*80)

  if not os.path.exists(PROJECTION_CACHE) or options.project:
    print('-'*80)
    print("Projecting image features into 3 dimensions using t-SNE...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    model = TSNE(n_components=3, random_state=0)
    projected = model.fit_transform(features)
    with open(PROJECTION_CACHE, 'wb') as fout:
      pickle.dump(projected, fout)
    print("  ...done projecting image features.")
    print('-'*80)
  else:
    print('-'*80)
    print("Loading cached image projection...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    with open(PROJECTION_CACHE, 'rb') as fin:
      projected = pickle.load(fin)
    print("  ...done loading image projection.")
    print('-'*80)

  if not os.path.exists(CLUSTER_CACHE) or options.cluster:
    print('-'*80)
    print("Clustering images using {}...".format(CLUSTERING_METHOD.__name__))
    if options.pause:
      input("  Ready to continue (press enter) > ")
    metric = "euclidean"
    print("  Using metric '{}'".format(metric))

    # Decide cluster input:
    print("  Using input '{}'".format(CLUSTER_INPUT))
    if CLUSTER_INPUT == "features":
      clin = features
    else:
      clin = projected

    # If we're not using DBSCAN we don't need clustering_distance
    if CLUSTERING_METHOD == DBSCAN:
      # Figure out what the distance value should be:
      print("  Computing nearest-neighbor distances...")

      distances = pairwise.pairwise_distances(clin, metric=metric)
      # sort our distance array and take the first few as nearby points
      # offset by 1 excludes the zero distance to self
      # TODO: Why doesn't this work?!?
      ordered_distances = np.sort(distances, axis=1)[:,1:DBSCAN_N_NEIGHBORS+1]
      outer_distances = ordered_distances[:,DBSCAN_N_NEIGHBORS-1]
      outer_distances = np.sort(outer_distances)
      smp = outer_distances[::N_EXAMPLES//10]
      print("   Distance sample:")
      print(smp)
      print("  ...done.")
      #closest, min_dist = pairwise.pairwise_distances_argmin_min(
      #  clin,
      #  clin,
      #  metric=metric
      #)
      clustering_distance = 0
      perc = DBSCAN_PERCENTILE
      while clustering_distance == 0 and perc < 100:
        clustering_distance = np.percentile(outer_distances, perc)
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

    #model = DBSCAN(metric=metric)
    if CLUSTERING_METHOD == DBSCAN:
      model = DBSCAN(
        eps=clustering_distance,
        min_samples=DBSCAN_N_NEIGHBORS,
        metric=metric,
        algorithm="auto"
      )
    elif CLUSTERING_METHOD == AffinityPropagation:
      model = CLUSTERING_METHOD(affinity=metric)
    elif CLUSTERING_METHOD == AgglomerativeClustering:
      model = CLUSTERING_METHOD(
        affinity=metric,
        linkage="average",
        n_clusters=2
      )
    # method "cluster" doesn't have a model

    if CLUSTERING_METHOD == NovelClustering:
      clusters = CLUSTERING_METHOD(clin)
    else:
      fit = model.fit(clin)
      clusters = fit.labels_

    if CLUSTERING_METHOD == DBSCAN:
      core_mask = np.zeros_like(fit.labels_, dtype=int)
      core_mask[fit.core_sample_indices_] = 1
      core_count = np.count_nonzero(core_mask)
      print(
        "Core samples: {}/{} ({:.2f}%)".format(
          core_count,
          N_EXAMPLES, 
          100 * core_count / N_EXAMPLES
        )
      )


    valid_clusters = set(clusters)
    unfiltered = len(valid_clusters)

    cluster_counts = count_clusters(clusters)

    for i in range(len(clusters)):
      if cluster_counts[clusters[i]] == 1:
        clusters[i] = -1

    valid_clusters = set(clusters)
    if len(valid_clusters) != unfiltered:
      # Have to reassign cluster IDs:
      remap = {}
      new_id = 0
      for i in range(len(clusters)):
        if clusters[i] == -1:
          continue
        if clusters[i] not in remap:
          remap[clusters[i]] = new_id
          clusters[i] = new_id
          new_id += 1
        else: 
          clusters[i] = remap[clusters[i]]

    valid_clusters = set(clusters)

    if -1 in valid_clusters:
      print(
        "  Found {} cluster(s) (with outliers)".format(len(valid_clusters) - 1)
      )
    else:
      print(
        "  Found {} cluster(s) (no outliers)".format(len(valid_clusters))
      )

    with open(CLUSTER_CACHE, 'wb') as fout:
      if CLUSTERING_METHOD == DBSCAN:
        pickle.dump(
          (ordered_distances, outer_distances, clusters, core_mask),
          fout
        )
      else:
        pickle.dump(clusters, fout)
    print("  ...done clustering images.")
    print('-'*80)
  else:
    print('-'*80)
    print("Loading cached clusters...")
    if options.pause:
      input("  Ready to continue (press enter) > ")
    with open(CLUSTER_CACHE, 'rb') as fin:
      if CLUSTERING_METHOD == DBSCAN:
        ordered_distances, outer_distances, clusters, core_mask = pickle.load(
          fin
        )
      else:
        clusters = pickle.load(fin)
    valid_clusters = set(clusters)
    if -1 in valid_clusters:
      print(
        "  Loaded {} cluster(s) (with outliers)".format(len(valid_clusters) - 1)
      )
    else:
      print(
        "  Loaded {} cluster(s) (no outliers)".format(len(valid_clusters))
      )
    print("  ...done loading clusters.")
    print('-'*80)

  # Plot a histogram of error values for all images:
  print("Plotting reconstruction error histogram...")
  if options.pause:
    input("  Ready to continue (press enter) > ")
  print("  Error limits:", min(ratings), max(ratings))
  plt.clf()
  n, bins, patches = plt.hist(ratings, 100)
  #plt.plot(bins)
  plt.xlabel("Mean Squared Error")
  plt.ylabel("Number of Images")
  #plt.axis([0, 1.1*max(ratings), 0, 1.2 * max(n)])
  #plt.show()
  plt.savefig(os.path.join(OUTPUT_DIR, HISTOGRAM_FILENAME.format("error")))
  plt.clf()
  print("  ...done.")

  # Plot the t-SNE results:
  print("Plotting t-SNE results...")
  if options.pause:
    input("  Ready to continue (press enter) > ")
  cycle = palettable.tableau.Tableau_20.mpl_colors
  colors = []
  alt_colors = []
  for c in classes:
    colors.append(cycle[c % len(cycle)])

  sizes = 0.25
  if CLUSTERING_METHOD == DBSCAN:
    sizes = core_mask*0.75 + 0.25

  for cl in clusters:
    if cl == -1:
      alt_colors.append((0.0, 0.0, 0.0)) # black
    else:
      alt_colors.append(cycle[cl % len(cycle)])

  for dims in [(0, 1), (0, 2), (1, 2)]:
    # Plot using true colors:
    x, y = dims
    plt.clf()
    ax = plt.scatter(projected[:,x], projected[:,y], s=0.25, c=colors)
    plt.xlabel("t-SNE {}".format("xyz"[x]))
    plt.ylabel("t-SNE {}".format("xyz"[y]))
    plt.savefig(os.path.join(OUTPUT_DIR, TSNE_FILENAME.format("true", x, y)))
    plt.clf()

    # Plot using guessed colors:
    x, y = dims
    plt.clf()
    ax = plt.scatter(projected[:,x], projected[:,y], s=sizes, c=alt_colors)
    plt.xlabel("t-SNE {}".format("xyz"[x]))
    plt.ylabel("t-SNE {}".format("xyz"[y]))
    plt.savefig(os.path.join(OUTPUT_DIR, TSNE_FILENAME.format("learned", x, y)))
    plt.clf()

  # TODO: Less hacky here
  montage_images(".", TSNE_FILENAME.format("*", "*", "{}"))
  print("  ...done.")

  # Plot a histogram of pairwise distance values (if we computed them):
  if CLUSTERING_METHOD == DBSCAN:
    print(
      "Plotting distance histograms...".format(DBSCAN_N_NEIGHBORS)
    )
    if options.pause:
      input("  Ready to continue (press enter) > ")

    for col in range(ordered_distances.shape[1]):
      #n, bins, patches = plt.hist(
      #  ordered_distances[:,col],
      #  1000,
      #  cumulative=True
      #)
      n, bins, patches = plt.hist(ordered_distances[:,col], 1000)
      #plt.plot(bins)
      plt.xlabel("Distance to {} Neighbor".format(ordinal(col+1)))
      plt.ylabel("Number of Images")
      #plt.axis([0, 1.1*max(outer_distances), 0, 1.2 * max(n)])
      plt.savefig(
        os.path.join(
          OUTPUT_DIR,
          HISTOGRAM_FILENAME.format("distance-{}".format(col))
        )
      )
      plt.clf()
    montage_images(".", HISTOGRAM_FILENAME.format("distance-{}"))
    print("  ...done.")

  plt.plot(outer_distances)
  plt.xlabel("Index")
  plt.ylabel("Distance to {} Neighbor".format(ordinal(DBSCAN_N_NEIGHBORS)))
  plt.savefig(
    os.path.join(
      OUTPUT_DIR,
      DISTANCE_FILENAME.format(ordinal(DBSCAN_N_NEIGHBORS))
    )
  )
  plt.clf()

  n, bins, patches = plt.hist(
    ordered_distances[:,1] / ordered_distances[:,0],
    1000
  )
  plt.xlabel("Distance ratio between 1st and 2nd Neighbors")
  plt.ylabel("Number of Images")
  plt.savefig(
    os.path.join(
      OUTPUT_DIR,
      HISTOGRAM_FILENAME.format("distance-ratio")
    )
  )
  plt.clf()

  # Plot cluster sizes
  print("Plotting cluster size histogram...")
  cluster_counts = count_clusters(clusters)
  just_counts = list(reversed(sorted(list(cluster_counts.values()))))
  small_counts = [c for c in just_counts if c < 50]
  large_counts = [c for c in just_counts if c >= 50]
  print("  Large cluster counts (not plotted):", large_counts)
  plt.clf()
  n, bins, patches = plt.hist(small_counts, 50)
  #plt.plot(bins)
  plt.xlabel("Images in Cluster")
  plt.ylabel("Number of Clusters")
  #plt.axis([0, max(small_counts), 0, 1.2 * max(n)])
  plt.savefig(os.path.join(OUTPUT_DIR, HISTOGRAM_FILENAME.format("clusters")))
  plt.clf()
  print("  ...done.")

  # Show some of the clustering results (TODO: better):
  print("Sampling clustered images...")
  if options.pause:
    input("  Ready to continue (press enter) > ")
  try:
    os.mkdir(os.path.join(OUTPUT_DIR, CLUSTERS_DIR), mode=0o755)
  except FileExistsError:
    pass

  # TODO: Get more representative images?
  nvc = min(MAX_CLUSTER_SAMPLES, len(valid_clusters))
  for i, c in enumerate(list(valid_clusters)[:MAX_CLUSTER_SAMPLES]):
    prbar(i / nvc)
    cluster_images = []
    shuf = list(zip(images, clusters))
    random.shuffle(shuf)
    for i, (img, cluster) in enumerate(shuf):
      if cluster == c:
        cluster_images.append(img)
        if len(cluster_images) >= SAMPLES_PER_CLUSTER:
          break
    if c == -1:
      thisdir = os.path.join(CLUSTERS_DIR, "outliers")
    else:
      thisdir = os.path.join(CLUSTERS_DIR, "cluster-{}".format(c))
    try:
      os.mkdir(os.path.join(OUTPUT_DIR, thisdir), mode=0o755)
    except FileExistsError:
      pass
    save_images(cluster_images, thisdir, CLUSTER_REP_FILENAME)
    montage_images(thisdir, CLUSTER_REP_FILENAME)

  print() # done with the progress bar
  collect_montages(CLUSTERS_DIR)
  print("  ...done.")

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option(
    "-m",
    "--model",
    action="store_true",
    help="Recompute the model even if a cached model is found."
  )
  parser.add_option(
    "-r",
    "--rank",
    action="store_true",
    help="Recompute rankings even if cached rankings are found."
  )
  parser.add_option(
    "-f",
    "--features",
    action="store_true",
    help="Recompute features even if cached features are found."
  )
  parser.add_option(
    "-j",
    "--project",
    action="store_true",
    help="Recompute t-SNE projection even if a cached projection is found."
  )
  parser.add_option(
    "-c",
    "--cluster",
    action="store_true",
    help="Recompute clusters even if a cached clustering is found."
  )
  parser.add_option(
    "-p",
    "--pause",
    action="store_true",
    help="Pause for user input at each step."
  )
  parser.add_option(
    "-F",
    "--fresh",
    action="store_true",
    help="Recompute everything."
  )
  options, args = parser.parse_args()
  if options.fresh:
    options.model = True
    options.rank = True
    options.features = True
    options.project = True
    options.cluster = True
  main(options)

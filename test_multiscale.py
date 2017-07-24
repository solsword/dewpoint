#!/usr/bin/env python3
"""
Tests for multiscale.py.
"""

import utils

import math
import warnings
import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc

from scipy.optimize import curve_fit

import multiscale

def gcluster(loc, size, n):
  return np.stack(
    [
      np.random.normal(l, s, size=n)
        for (l, s) in zip(loc, size)
    ],
    axis=-1
  )

def gring(loc, size, width, n):
  functions = []
  for i in range(len(size)):
    functions.append([np.cos, np.sin][i%2])

  return np.stack(
    [
      l + (
        s + np.random.normal(0, width, n)
      ) * func(np.linspace(0, 2*math.pi, n))
        for (l, s, func) in zip(loc, size, functions)
    ],
    axis=-1
  )

def gline(fr, to, var, n):
  return np.concatenate(
    [
      np.linspace(fr[0], to[0], n).reshape((n, 1)),
      np.linspace(fr[1], to[1], n).reshape((n, 1))
    ],
    axis=1
  ) + gcluster((0.0, 0.0), (var, var), n)

test_cases = [
  np.random.uniform(0, 1, size=(100, 2)),
  gcluster((0.5, 0.5), (0.3, 0.3), 100),
  gcluster((0.5, 0.5), (1.0, 1.0), 100),
  gring((0.5, 0.5), (0.4, 0.6), 0.03, 100),
  np.concatenate(
    [
      gcluster((0.2, 0.6), (0.05, 0.05), 100),
      gcluster((0.7, 0.3), (0.07, 0.07), 100),
    ]
  ),
  np.concatenate(
    [
      gring((0.2, 0.6), (0.2, 0.15), 0.05, 100),
      gring((0.7, 0.3), (0.1, 0.3), 0.05, 100),
    ]
  ),
  np.concatenate(
    [
      gcluster((0.2, 0.2), (0.03, 0.03), 100),
      gcluster((0.1, 0.3), (0.03, 0.03), 100),
      gcluster((0.9, 0.4), (0.08, 0.08), 100),
    ]
  ),
  np.concatenate(
    [
      gcluster((0.2, 0.2), (0.03, 0.03), 100),
      gcluster((0.1, 0.8), (0.03, 0.03), 100),
      gcluster((0.9, 0.4), (0.08, 0.08), 100),
    ]
  ),
  np.concatenate(
    [
      gcluster((0.2, 0.2), (0.03, 0.03), 100),
      gcluster((0.1, 0.8), (0.03, 0.03), 100),
      gcluster((0.6, 0.4), (0.08, 0.12), 100),
      gcluster((0.9, 0.9), (0.15, 0.15), 100),
    ]
  ),
  np.concatenate(
    [
      gring((0.5, 0.5), (0.6, 0.6), 0.04, 100),
      gring((0.5, 0.5), (0.2, 0.2), 0.05, 100),
    ]
  ),
  np.concatenate(
    [
      gcluster((0.2, 0.2), (0.06, 0.06), 100),
      gcluster((0.8, 0.8), (0.04, 0.04), 100),
      gcluster((0.2, 0.8), (0.06, 0.04), 100),
      gcluster((0.8, 0.2), (0.04, 0.06), 100),
      gcluster((0.5, 0.5), (0.6, 0.6), 300),
      gring((0.45, 0.55), (0.9, 0.9), 0.07, 200),
    ]
  ),
  np.concatenate(
    [
      gcluster((0.2, 0.6), (0.05, 0.05), 100),
      gcluster((0.7, 0.3), (0.07, 0.07), 100),
      gline((0, 0), (1, 1), 0.015, 50)
    ],
    axis=0
  ),
  np.concatenate(
    [
      gcluster((0.5, 0.5), (0.5, 0.5), 100),
      gcluster((0.7, 0.3), (0.3, 0.5), 100),
      gcluster((0.3, 0.4), (0.1, 0.08), 100),
      gcluster((1.3, 1.5), (0.4, 0.1), 100)
    ],
    axis=0
  ),
  np.concatenate(
    [
      gline((0.1, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.9, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.3, 0.5), (0.7, 0.5), 0.02, 100),
    ],
    axis=0
  ),
  np.concatenate(
    [
      gline((0.1, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.9, 0.1), (0.5, 0.9), 0.02, 100),
      gline((0.3, 0.5), (0.7, 0.5), 0.02, 100),
      gline((1.1, 0.1), (1.1, 0.9), 0.02, 100),
      gring((1.3, 0.3), (0.2, 0.2), 0.01, 100),
    ],
    axis=0
  ),
  np.concatenate(
    [
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50),
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(1, 0, -0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50)
    ],
    axis=0
  ),
  np.concatenate(
    [
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50),
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1) + 0.4
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50)
    ],
    axis=0
  ),
  np.concatenate( # a 3D triple helix
    (
      np.stack(
        (
          np.cos(np.linspace(0, 2 * math.pi, 100)),
          np.linspace(0, 2 * math.pi, 100),
          np.sin(np.linspace(0, 2 * math.pi, 100)),
        ),
        axis=-1
      ) + gcluster((0, 0, 0), (0.18, 0.09, 0.18), 100),
      np.stack(
        (
          np.cos(2 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
          np.linspace(0, 2 * math.pi, 100),
          np.sin(2 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
        ),
        axis=-1
      ) + gcluster((0, 0, 0), (0.18, 0.09, 0.18), 100),
      np.stack(
        (
          np.cos(4 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
          np.linspace(0, 2 * math.pi, 100),
          np.sin(4 * math.pi / 3 + np.linspace(0, 2 * math.pi, 100)),
        ),
        axis=-1
      ) + gcluster((0, 0, 0), (0.18, 0.09, 0.18), 100),
    ),
    axis=0
  )
]

KERNEL = np.asarray([0.75, 0.75, 1, 1, 1, 1, 1, 0.75, 0.75])
KERNEL /= sum(KERNEL)

PREV_KERNEL = np.asarray([0.0]*5 + [1.0]*6)
PREV_KERNEL /= sum(PREV_KERNEL)

def newfig():
  utils.reset_color()
  plt.figure()

def plot_data(
  data,
  mute=False,
  label=None,
  smooth=False,
  baseline=False,
  stats=False,
):
  c, d = utils.pick_color(both=True)
  if mute:
    c = d
    d = (0.4, 0.4, 0.4)

  plt.axhline(0, color=(0, 0, 0))
  plt.plot(
    data,
    color=c,
    label=label
  )
  if baseline:
    base = np.median(data[:5,:])*baseline
    plt.axhline(base, color=d)

  if smooth:
    thresh = np.convolve(
      data,
      PREV_KERNEL,
      mode="same"
    ) * smooth
    plt.plot(thresh, color=d)

  if stats:
    mean = np.mean(data)
    std = np.std(data)
    plt.axhline(mean, color=d)
    plt.axhline(mean + std, color=d)
    plt.axhline(mean - std, color=d)

def plot_property(points, properties, ax=None, title=None):
  print("Plotting {} points..".format(len(points)))
  projected = points[:,:2]

  if ax == None:
    fig, ax = plt.subplots()
    if title:
      plt.title(title)
    else:
      plt.title("Points")
    plt.axis("equal")
  elif title:
    ax.set_title(title)

  cmap = plt.get_cmap("plasma")
  colors = np.apply_along_axis(cmap, 0, properties)

  ax.scatter(projected[:,0], projected[:,1], s=1.0, c=colors)

  print("...done plotting points.")

def plot_clusters(points, clusters, ax=None, title=None, show_quality=None):
  print("Plotting {} clusters..".format(len(clusters)))

  # Plot in 2 dimensions; ignore any other dimensions of the data
  projected = points[:,:2]

  if ax == None:
    fig, ax = plt.subplots()
    if title:
      plt.title(title)
    else:
      plt.title("Clustering Results")
    plt.axis("equal")
  elif title:
    ax.set_title(title)


  ax.scatter(projected[:,0], projected[:,1], s=0.8, c=utils.POINT_COLOR)

  by_size = sorted(list(clusters.items()), key=lambda kv: -kv[1]["size"])

  cmap = {}
  for ck, cl in by_size:
    if ck not in cmap:
      clcolor = utils.pick_color()
      cmap[ck] = clcolor
    else:
      clcolor = cmap[ck]

    edges = []
    colors = []
    widths = []
    styles = []

    if not cl["edges"]:
      continue

    for (fr, to, d, q) in cl["edges"]:
      edges.append((projected[fr], projected[to]))
      colors.append(clcolor)
      if show_quality == "edges":
        widths.append(2*q)
      elif show_quality in cl:
        widths.append(2*cl[show_quality])
      else:
        widths.append(0.8)
      if fr in cl["outliers"] or to in cl["outliers"]:
        styles.append("dotted")
      else:
        styles.append("solid")

    lc = mc.LineCollection(
      edges,
      colors=colors,
      linewidths=widths,
      linestyles=styles
    )
    ax.add_collection(lc)
  print("...done plotting results.")


def plot_stats(clusters, stats, sort_by="size", normalize=None, show_mean=None):

  collected = {k: [] for k in stats}

  n = len(clusters)

  ordered = sorted(list(clusters.values()), key=lambda cl: cl[sort_by])

  for cl in ordered:
    for st in stats:
      if st in cl:
        collected[st].append(cl[st])
      else:
        collected[st].append(-0.1)

  for st in stats:
    collected[st] = np.asarray(collected[st])
    if normalize and st in normalize:
      upper = np.max(collected[st])
      collected[st] = collected[st] / upper

  plt.figure()
  utils.reset_color()
  for st in stats:
    if show_mean and st in show_mean:
      cp, cs = utils.pick_color(both=True)
      plt.scatter(
        range(n),
        collected[st],
        label=st,
        c=cp,
        s=0.8
      )
      plt.axhline(
        np.mean(collected[st]),
        label="mean_" + st,
        c=cs,
        lw=0.6
      )
    else:
      plt.scatter(
        range(n),
        collected[st],
        label=st,
        c=utils.pick_color(),
        s=0.8
      )

  plt.legend()
  plt.xlabel("Cluster Stats")

def cluster_sequence(clusters):
  root = multiscale.find_largest(clusters)

  seq = [ { root["id"]: root } ]

  while any(cl["children"] for cl in seq[-1].values()):
    seq.append({})
    for cl in seq[-2].values():
      for child in cl["children"]:
        seq[-1][child["id"]] = child

  return seq

def plot_cluster_sequence(points, clusters):
  seq = cluster_sequence(clusters)
  sqw = int(math.ceil(len(seq)**0.5))
  sqh = sqw
  while len(seq) <= sqw * sqh - sqw:
    sqh -= 1
  if sqw * sqh == 1:
    plot_clusters(points, clusters)
  else:
    fig, axes = plt.subplots(sqh, sqw, sharex=True, sharey=True)
    plt.axis("equal")
    for i, cls in enumerate(seq):
      sqx, sqy = i % sqw, i // sqw
      if sqh == 1:
        plot_clusters(points, cls, ax=axes[sqx])
      else:
        plot_clusters(points, cls, ax=axes[sqy][sqx])

IRIS_DATA = np.asarray([
  [5.1,3.5,1.4,0.2],
  [4.9,3.0,1.4,0.2],
  [4.7,3.2,1.3,0.2],
  [4.6,3.1,1.5,0.2],
  [5.0,3.6,1.4,0.2],
  [5.4,3.9,1.7,0.4],
  [4.6,3.4,1.4,0.3],
  [5.0,3.4,1.5,0.2],
  [4.4,2.9,1.4,0.2],
  [4.9,3.1,1.5,0.1],
  [5.4,3.7,1.5,0.2],
  [4.8,3.4,1.6,0.2],
  [4.8,3.0,1.4,0.1],
  [4.3,3.0,1.1,0.1],
  [5.8,4.0,1.2,0.2],
  [5.7,4.4,1.5,0.4],
  [5.4,3.9,1.3,0.4],
  [5.1,3.5,1.4,0.3],
  [5.7,3.8,1.7,0.3],
  [5.1,3.8,1.5,0.3],
  [5.4,3.4,1.7,0.2],
  [5.1,3.7,1.5,0.4],
  [4.6,3.6,1.0,0.2],
  [5.1,3.3,1.7,0.5],
  [4.8,3.4,1.9,0.2],
  [5.0,3.0,1.6,0.2],
  [5.0,3.4,1.6,0.4],
  [5.2,3.5,1.5,0.2],
  [5.2,3.4,1.4,0.2],
  [4.7,3.2,1.6,0.2],
  [4.8,3.1,1.6,0.2],
  [5.4,3.4,1.5,0.4],
  [5.2,4.1,1.5,0.1],
  [5.5,4.2,1.4,0.2],
  [4.9,3.1,1.5,0.1],
  [5.0,3.2,1.2,0.2],
  [5.5,3.5,1.3,0.2],
  [4.9,3.1,1.5,0.1],
  [4.4,3.0,1.3,0.2],
  [5.1,3.4,1.5,0.2],
  [5.0,3.5,1.3,0.3],
  [4.5,2.3,1.3,0.3],
  [4.4,3.2,1.3,0.2],
  [5.0,3.5,1.6,0.6],
  [5.1,3.8,1.9,0.4],
  [4.8,3.0,1.4,0.3],
  [5.1,3.8,1.6,0.2],
  [4.6,3.2,1.4,0.2],
  [5.3,3.7,1.5,0.2],
  [5.0,3.3,1.4,0.2],
  [7.0,3.2,4.7,1.4],
  [6.4,3.2,4.5,1.5],
  [6.9,3.1,4.9,1.5],
  [5.5,2.3,4.0,1.3],
  [6.5,2.8,4.6,1.5],
  [5.7,2.8,4.5,1.3],
  [6.3,3.3,4.7,1.6],
  [4.9,2.4,3.3,1.0],
  [6.6,2.9,4.6,1.3],
  [5.2,2.7,3.9,1.4],
  [5.0,2.0,3.5,1.0],
  [5.9,3.0,4.2,1.5],
  [6.0,2.2,4.0,1.0],
  [6.1,2.9,4.7,1.4],
  [5.6,2.9,3.6,1.3],
  [6.7,3.1,4.4,1.4],
  [5.6,3.0,4.5,1.5],
  [5.8,2.7,4.1,1.0],
  [6.2,2.2,4.5,1.5],
  [5.6,2.5,3.9,1.1],
  [5.9,3.2,4.8,1.8],
  [6.1,2.8,4.0,1.3],
  [6.3,2.5,4.9,1.5],
  [6.1,2.8,4.7,1.2],
  [6.4,2.9,4.3,1.3],
  [6.6,3.0,4.4,1.4],
  [6.8,2.8,4.8,1.4],
  [6.7,3.0,5.0,1.7],
  [6.0,2.9,4.5,1.5],
  [5.7,2.6,3.5,1.0],
  [5.5,2.4,3.8,1.1],
  [5.5,2.4,3.7,1.0],
  [5.8,2.7,3.9,1.2],
  [6.0,2.7,5.1,1.6],
  [5.4,3.0,4.5,1.5],
  [6.0,3.4,4.5,1.6],
  [6.7,3.1,4.7,1.5],
  [6.3,2.3,4.4,1.3],
  [5.6,3.0,4.1,1.3],
  [5.5,2.5,4.0,1.3],
  [5.5,2.6,4.4,1.2],
  [6.1,3.0,4.6,1.4],
  [5.8,2.6,4.0,1.2],
  [5.0,2.3,3.3,1.0],
  [5.6,2.7,4.2,1.3],
  [5.7,3.0,4.2,1.2],
  [5.7,2.9,4.2,1.3],
  [6.2,2.9,4.3,1.3],
  [5.1,2.5,3.0,1.1],
  [5.7,2.8,4.1,1.3],
  [6.3,3.3,6.0,2.5],
  [5.8,2.7,5.1,1.9],
  [7.1,3.0,5.9,2.1],
  [6.3,2.9,5.6,1.8],
  [6.5,3.0,5.8,2.2],
  [7.6,3.0,6.6,2.1],
  [4.9,2.5,4.5,1.7],
  [7.3,2.9,6.3,1.8],
  [6.7,2.5,5.8,1.8],
  [7.2,3.6,6.1,2.5],
  [6.5,3.2,5.1,2.0],
  [6.4,2.7,5.3,1.9],
  [6.8,3.0,5.5,2.1],
  [5.7,2.5,5.0,2.0],
  [5.8,2.8,5.1,2.4],
  [6.4,3.2,5.3,2.3],
  [6.5,3.0,5.5,1.8],
  [7.7,3.8,6.7,2.2],
  [7.7,2.6,6.9,2.3],
  [6.0,2.2,5.0,1.5],
  [6.9,3.2,5.7,2.3],
  [5.6,2.8,4.9,2.0],
  [7.7,2.8,6.7,2.0],
  [6.3,2.7,4.9,1.8],
  [6.7,3.3,5.7,2.1],
  [7.2,3.2,6.0,1.8],
  [6.2,2.8,4.8,1.8],
  [6.1,3.0,4.9,1.8],
  [6.4,2.8,5.6,2.1],
  [7.2,3.0,5.8,1.6],
  [7.4,2.8,6.1,1.9],
  [7.9,3.8,6.4,2.0],
  [6.4,2.8,5.6,2.2],
  [6.3,2.8,5.1,1.5],
  [6.1,2.6,5.6,1.4],
  [7.7,3.0,6.1,2.3],
  [6.3,3.4,5.6,2.4],
  [6.4,3.1,5.5,1.8],
  [6.0,3.0,4.8,1.8],
  [6.9,3.1,5.4,2.1],
  [6.7,3.1,5.6,2.4],
  [6.9,3.1,5.1,2.3],
  [5.8,2.7,5.1,1.9],
  [6.8,3.2,5.9,2.3],
  [6.7,3.3,5.7,2.5],
  [6.7,3.0,5.2,2.3],
  [6.3,2.5,5.0,1.9],
  [6.5,3.0,5.2,2.0],
  [6.2,3.4,5.4,2.3],
  [5.9,3.0,5.1,1.8],
])

IRIS_LABELS = [
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-setosa",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-versicolor",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
  "Iris-virginica",
]

def test_typicality():
  with open("cache/cached-features.pkl", 'rb') as fin:
    features = pickle.load(fin)
  with open("cache/cached-projection.pkl", 'rb') as fin:
    proj = pickle.load(fin)
  for tc, pr in [(features, proj)]:
    typ = multiscale.typicality(tc, quiet=False)
    plot_property(pr, typ, title="Typicality")
    plt.show()

def project(x):
  return x[:,:2]

def test():
  #stuff = test_cases[:8]
  #stuff = test_cases[-2:]
  #stuff = test_cases
  #stuff = zip(stuff, [project(x) for x in stuff])

  with open("cache/cached-features.pkl", 'rb') as fin:
    features = pickle.load(fin)
  with open("cache/cached-projection.pkl", 'rb') as fin:
    proj = pickle.load(fin)
  stuff = [(features, proj)]
  #stuff = [(features, project(features))]

  for tc, pr in stuff:
    # Cluster projected rather than real values:
    #tc = pr
    print("Finding nearest neighbors to generate edge set...")
    nbd, nbi = multiscale.neighbors_from_points(tc, 20, "euclidean")
    print("...done.")
    print("Generating edges from nearest neighbors...")
    edges = multiscale.get_neighbor_edges(nbd, nbi)
    print("...done.")
    clusters = multiscale.multiscale_clusters(tc, edges=edges, quiet=False)

    #root = multiscale.find_largest(clusters)
    #plot_clusters(tc, {0: root}, title="Root Cluster", show_quality="edges")

    print("Condensing best clusters...")
    top = multiscale.condense_best(
      clusters,
      threshold=0.95,
      quality="quality"
    )
    print("...found {} condensed clusters.".format(len(top)))
    print("Retaining only large clusters...")
    top = multiscale.retain_above(
      top,
      threshold=10,
      filter_on="size"
    )
    print("...found {} large clusters.".format(len(top)))
    #top = multiscale.retain_best(clusters, filter_on="mixed_quality")
    #top = multiscale.retain_best(top, filter_on="mixed_quality")
    #top = multiscale.retain_best(clusters, filter_on="compactness")
    #top = multiscale.retain_best(top, filter_on="separation")

    #sep = multiscale.decant_split(
    #  clusters,
    #  #top,
    #  threshold=1.0,
    #  #criterion=multiscale.quality_vs_coverage_criterion(
    #  #  size="core_size",
    #  #  quality="quality"
    #  #)
    #  #criterion=lambda cl: cl["size"] / (len(tc)/4)
    #  #threshold=0.9,
    #  #criterion=multiscale.generational_criterion(key="quality")
    #  #criterion=multiscale.generational_criterion(feature="separation")
    #  criterion=multiscale.product_criterion(
    #    multiscale.generational_criterion(feature="separation"),
    #    multiscale.generational_criterion(feature="quality")
    #  )
    #  #criterion=multiscale.quality_vs_outliers_criterion(quality="quality")
    #  #criterion=multiscale.satisfaction_criterion(quality="quality")
    #  #criterion=multiscale.scale_quality_criterion(quality="quality")
    #)
    print("Decanting best clusters...")
    sep = multiscale.decant_best(top, quality="quality")
    print("...retained {} best separate clusters.".format(len(sep)))
    #sep = multiscale.decant_erode(tc, clusters, threshold=0.6)
    #best = multiscale.vote_refine(tc, clusters, quality="quality")

    utils.reset_color()
    plot_stats(
      clusters,
      [
        "mean",
        "size",
        "scale",
        "quality",
        "obviousness",
        "mixed_quality",
        "split_quality"
      ],
      #sort_by="size",
      #sort_by="scale",
      sort_by="mean",
      normalize=["mean", "size", "scale"],
      show_mean=["mixed_quality"],
    )

    #utils.reset_color()
    #plot_cluster_sequence(pr, clusters)

    #utils.reset_color()
    #plot_clusters(pr, clusters, title="All {} Clusters".format(len(clusters)))

    utils.reset_color()
    plot_clusters(pr, top, title="{} Filtered Cluster(s)".format(len(top)))

    utils.reset_color()
    plot_clusters(pr, sep, title="{} Split Cluster(s)".format(len(sep)))

    #utils.reset_color()
    #plot_clusters(pr, best, title="{} Revised Cluster(s)".format(len(best)))

    #analyze_clustering(points, neighbors, top)

    plt.show()

  #clusters = multiscale.multiscale_clusters(IRIS_DATA)
  #top = clusters
  ##top = multiscale.retain_best(clusters)
  #plot_clusters(IRIS_DATA, top)
  #plot_stats(top)
  #plt.show()

def run_strict(f, *args, **kwargs):
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    f(*args, **kwargs)

if __name__ == "__main__":
  #test()
  run_strict(test)
  #test_typicality()

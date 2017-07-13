#!/usr/bin/env python3
"""
Tests for multiscale.py.
"""

import utils

import math

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc

import multiscale

PLOT = [
  #"averages",
  #"distance_space",
  #"std_diff",
  #"local_linearity",
  #"neighbor_counts",
  #"impact",
  #"impact_ratio",
  #"significance",
  #"local_structure",
  #"normalized_distances",
  #"coherence",
  #"local",
  #"lcompare",
  #"cut",
  #"quartiles",
  #"edge_lengths",
  #"included_lengths",
  #"raw",
  #"absolute_growth",
  "results",
  #"result_stats",
]

def gcluster(loc, size, n):
  return np.concatenate(
    [
      np.random.normal(loc[0], size[0], size=n).reshape((n,1)),
      np.random.normal(loc[1], size[1], size=n).reshape((n,1))
    ],
    axis=1
  )

def gring(loc, size, width, n):
  steps = np.arange(0, 2*math.pi, 2*math.pi / n)
  r = (size[0] + np.random.normal(0, width, size=n))
  xs = loc[0] + np.multiply(r, np.cos(steps))
  r = (size[1] + np.random.normal(0, width, size=n))
  ys = loc[1] + np.multiply(r, np.sin(steps))
  return np.concatenate([xs.reshape((n,1)), ys.reshape((n,1))], axis=1)

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
      gcluster((0.2, 0.6), (0.05, 0.05), 100),
      gcluster((0.7, 0.3), (0.07, 0.07), 100),
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


def plot_clusters(points, clusters):
  print("Plotting {} clusters..".format(len(clusters)))

  # Plot in 2 dimensions; ignore any other dimensions of the data
  projected = points[:,:2]

  fig, ax = plt.subplots()
  plt.title("Clustering Results")
  plt.axis("equal")
  plt.scatter(projected[:,0], projected[:,1], s=0.8, c=utils.POINT_COLOR)
  cmap = {}
  for ck in clusters:
    if ck not in cmap:
      clcolor = utils.pick_color()
      cmap[ck] = clcolor
    else:
      clcolor = cmap[ck]

    cl = clusters[ck]
    edges = []
    colors = []
    widths = []
    for (fr, to, d, qc) in cl["edges"]:
      edges.append((projected[fr], projected[to]))
      if qc >= -0.1:
        colors.append(clcolor)
        widths.append(0.8)
      else:
        colors.append((1, 0, 1))
        widths.append(1.0 - 10 * qc)
    lc = mc.LineCollection(
      edges,
      colors=colors,
      linewidths=widths
    )
    ax.add_collection(lc)
  print("...done plotting results.")


def plot_stats(clusters, normalize=True):
  sizes = []
  scales = []
  qualities = []
  n = len(clusters)
  for ck in clusters:
    cl = clusters[ck]
    sizes.append(cl["size"])
    scales.append(cl["mean"])
    qualities.append(multiscale.cluster_quality(cl))

  sizes = np.asarray(list(reversed(sizes)))
  if normalize:
    sizes = sizes / np.max(sizes)

  scales = np.asarray(list(reversed(scales)))
  if normalize:
    scales = scales / np.max(scales)

  qualities = list(reversed(qualities))

  plt.figure()
  plt.scatter(range(n), sizes, label="size")
  plt.scatter(range(n), scales, label="scale")
  plt.scatter(range(n), qualities, label="quality")
  plt.legend()
  plt.xlabel("Cluster Stats")

  plt.show()

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

def test():
  #for tc in test_cases[4:6]:
  #for tc in test_cases[3:4]:
  #for tc in test_cases[4:9]:
  #for tc in test_cases[6:]:
  for tc in test_cases:
    clusters, assignments = multiscale.multiscale_clusters(tc, quiet=False)
    plot_clusters(tc, clusters)
    #plot_stats(clusters)
    #analyze_clustering(points, neighbors, best)
    plt.show()
  #multiscale.multiscale_clusters(IRIS_DATA)

if __name__ == "__main__":
  test()

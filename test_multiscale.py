#!/usr/bin/env python3
"""
Tests for multiscale.py.
"""

import numpy as np

import unionfind as uf

import matplotlib.pyplot as plt
from matplotlib import collections as mc

from utils import reset_color, pick_color, prbar

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
      gring((0.2, 0.6), (0.05, 0.05), 0.03, 100),
      gring((0.7, 0.3), (0.07, 0.07), 0.03, 100),
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
      np.concatenate(
        [
          np.arange(0, 1, 0.02).reshape(50, 1),
          np.arange(0, 1, 0.02).reshape(50, 1)
        ],
        axis=1
      ) + gcluster((0.0, 0.0), (0.012, 0.012), 50)
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

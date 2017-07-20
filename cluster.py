#!/usr/bin/env python3
"""
cluster.py
Implements a modified minimum-spanning-tree-based clustering algorithm given a
distance matrix.
"""

import math

import numpy as np

import unionfind as uf

# TODO: Testing:
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path as graph_shortest_paths
from scipy.stats import linregress
from scipy.spatial.distance import euclidean

from simplex_grid.simplex_grid import simplex_grid

COLORS = [
  (0.0, 0.2, 0.8),
  (0.8, 0.0, 0.0),
  (1.0, 0.7, 0.0),
  (0.0, 0.8, 0.1),
  (0.7, 0.3, 0.8),
  (0.0, 0.7, 0.7),
]
DESAT = [[(ch*2 + (sum(c) / len(c)))/3.5 for ch in c] for c in COLORS]

MIN_SIZE = 3
NEIGHBORHOOD_SIZE = 4
ANALYSIS_SIZE = 10

IMPACT_SIG_SIZE = 0.1
MIN_SIG_IMPACT = 2.5

SIG_THRESHOLD = 1.4

N_LARGEST = 5

#OUTLIER_CRITERION = 1.0
OUTLIER_CRITERION = 1.5

PREVENT_ZERO_MEANS = False
EPSILON = 0.00000000001

POINT_COLOR = (0, 0, 0)

# Whether or not to normalize distances
NORMALIZE_DISTANCES = True
NORMALIZATION_NEIGHBORHOOD = 4
#NORMALIZATION_STRENGTH = 0.25
NORMALIZATION_STRENGTH = 1.0

MANIFOLD_ERROR_THRESHOLD = 0.02
MANIFOLD_GRID_RESOLUTION = 10
MANIFOLD_EDGE_SAMPLES = 100
MANIFOLD_INTERIOR_SCALE_PERCENTILE = 75
MANIFOLD_APPROX_SCALE_MULTIPLIER = 1.5
MANIFOLD_MIN_NEIGHBORHOOD = 3


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
  "internal_distances",
  #"included_lengths",
  #"raw",
  #"absolute_growth",
  "results",
  #"result_stats",
  #"manifolds"
]

# TODO: Testing
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

def combine_info(A, B, e):
  fr, to, d = e
  sA = A["size"]
  sB = B["size"]
  mA = A["mean"]
  mB = B["mean"]
  vA = A["variance"]
  vB = B["variance"]
  pvA = vA * (sA - 1)
  pvB = vB * (sB - 1)

  if sA == 0 and sB == 0:
    return {
      "size": 1,
      "vertices": A["vertices"] | B["vertices"],
      "edges": { (fr, to, d) },
      "mean": d,
      "variance": 0,
      "largest": [d],
      "coherence": 0,
      "internal": d,
    }
  elif sA == 0 or sB == 0:
    nv = max(vA, vB)
    delta = d - max(mA, mB)
    rs = max(sA, sB)
    return {
      "size": sA + sB + 1,
      "vertices": A["vertices"] | B["vertices"],
      "edges": A["edges"] | B["edges"] | { (fr, to, d) },
      "mean": (max(sA * mA, sB * mB) + d) / (1 + rs),
      "variance": (
        nv
        + (
          delta**2 * rs
        ) / (rs + 1)
      ) / rs,
      "largest": sorted(A["largest"] + B["largest"] + [d])[-N_LARGEST:],
      "coherence": A["coherence"] + B["coherence"],
      "internal": max(A["internal"], B["internal"]) + max(sA + 1, sB + 1) * d,
    }
  else:
    result = {}
    result["size"] = sA + sB + 1
    result["vertices"] = A["vertices"] | B["vertices"]
    result["edges"] = A["edges"] | B["edges"] | { e }
    result["largest"] = sorted(A["largest"] + B["largest"] + [d])[-N_LARGEST:]
    result["coherence"] = A["coherence"] + B["coherence"]
    result["internal"] = A["internal"] + B["internal"] + d * (sA * sB)

    result["mean"] = (mA * sA + mB * sB + d) / (sA + sB + 1)
    if PREVENT_ZERO_MEANS:
      result["mean"] = max(EPSILON, result["mean"])

    # Incremental variance update from:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    delta = mA - mB
    pvA = vA * (sA - 1)
    pvB = vB * (sB - 1)
    result["variance"] = (
      pvA
    + pvB
    + (
        delta**2 * sA * sB
      ) / (sA + sB)
    ) / (sA + sB - 1)
    # Second combine step with the single edge:
    delta = (mA * sA + mB * sB) / (sA + sB) - d
    sR = result["size"]
    pvR = result["variance"] * (sR - 1)
    result["variance"] = (
      pvR + (
        delta**2 * sR
      ) / (sR + 1)
    ) / sR
  return result

PR_INTRINSIC = 0
PR_CHARS = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
def prbar(progress):
  global PR_INTRINSIC
  pbwidth = 65
  sofar = int(pbwidth * progress)
  left = pbwidth - sofar - 1
  ic = PR_CHARS[PR_INTRINSIC]
  PR_INTRINSIC = (PR_INTRINSIC + 1) % len(PR_CHARS)
  print("\r[" + "="*sofar + ">" + "-"*left + "] (" + ic + ")", end="")

KERNEL = np.asarray([0.75, 0.75, 1, 1, 1, 1, 1, 0.75, 0.75])
KERNEL /= sum(KERNEL)

PREV_KERNEL = np.asarray([0.0]*5 + [1.0]*6)
PREV_KERNEL /= sum(PREV_KERNEL)

CURRENT_COLOR = 0

def newfig():
  global CURRENT_COLOR
  plt.figure()
  CURRENT_COLOR = 0

def plot_data(
  data,
  desat=False,
  label=None,
  smooth=False,
  baseline=False,
  stats=False,
):
  global CURRENT_COLOR
  c = COLORS[CURRENT_COLOR % len(COLORS)]
  d = DESAT[CURRENT_COLOR % len(DESAT)]
  if desat:
    c = d
    d = (0.4, 0.4, 0.4)
  CURRENT_COLOR += 1

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

def cluster(points, metric="euclidean"):
  print("Starting clustering process...")
  print("  ...computing pairwise distances...")
  distances = pairwise.pairwise_distances(points, metric=metric)
  print("  ...done.")
  print("  ...computing {}-normalized distances...".format(NEIGHBORHOOD_SIZE))
  nearest = np.sort(distances, axis=1)[:,1:NORMALIZATION_NEIGHBORHOOD+1]
  neighbors = NearestNeighbors(
    n_neighbors=NORMALIZATION_NEIGHBORHOOD,
    algorithm="ball_tree"
  ).fit(points)
  norm_distances, norm_indices = neighbors.kneighbors(points)
  local_scale = np.median(norm_distances, axis=1)
  normalized = np.zeros_like(distances)
  for i in range(distances.shape[0]):
    for j in range(distances.shape[1]):
      normalized[i,j] = distances[i,j] / ((local_scale[i] + local_scale[j])/2)

  if NORMALIZE_DISTANCES:
    distances = (
      (normalized * NORMALIZATION_STRENGTH)
    + (distances * (1 - NORMALIZATION_STRENGTH))
    )
  print("  ...done.")
  print("  ...computing {} nearest neighbors...".format(NEIGHBORHOOD_SIZE))
  # Note: can't rely on kNN for Kruskal's algorithm because some edges are DQed.
  neighbors = NearestNeighbors(
    n_neighbors=NEIGHBORHOOD_SIZE,
    algorithm="ball_tree"
  ).fit(points)
  nbd, nbi = neighbors.kneighbors(points)
  print("  ...done.")

  projected = points[:,:2]

  n = distances.shape[0]
  edges = []
  longest = np.max(np.sort(distances, axis=1)[:,1])
  # TODO: Debug
  for fr in range(n):
    for to in range(fr+1, n):
      edges.append((fr, to, distances[fr,to], normalized[fr,to]))
      # TODO: non-symmetric metrics?
      #edges.append((to, fr, distances[to,fr], normalized[to,fr]))

  sorted_edges = sorted(edges, key=lambda e: (e[2], e[3]))
  u = uf.unionfind(distances.shape[0])

  prev_d = 0
  growth_rate = []
  local_growth_rate = []
  local_linearity = []
  average_cluster_size = []
  number_of_clusters = []
  coherence = []
  std_diff = []
  size_diff = []
  neighbor_counts = []
  local_structures = []
  normalized_distances = []
  impacts = []
  impact_ratios = []
  total_sizes = []
  included = []
  internal_distances = []
  clinfo = {}
  for i in range(n):
    clinfo[i] = {
      "size": 0,
      "vertices": { i },
      "edges": set(),
      "mean": 0,
      "variance": 0,
      "largest": [0] * N_LARGEST,
      "coherence": 0,
      "internal": 0,
    }
  print("  ...constructing MST...")
  total_points_clustered = 0
  for i, (fr, to, d, nr) in enumerate(sorted_edges):
    prbar(i / len(sorted_edges))
    r1 = u.find(fr)
    r2 = u.find(to)
    if r1 == r2:
      # increase coherence
      clinfo[r1]["coherence"] += 1
    else:
      i1 = clinfo[r1]
      i2 = clinfo[r2]
      i1s = i1["size"]
      i2s = i2["size"]
      ni = combine_info(i1, i2, (fr, to, d))
      del clinfo[r1]
      del clinfo[r2]

      total_points_clustered += i1s == 0
      total_points_clustered += i2s == 0

      included.append((fr, to, d, nr))
      u.unite(fr, to)
      clinfo[u.find(fr)] = ni

      # compute points affected
      total_sizes.append(total_points_clustered)
      absorbed = min((i1s + 1), (i2s + 1))
      dominant = max((i1s + 1), (i2s + 1))
      impacts.append(absorbed / total_points_clustered)
      # TODO: Less size-biased correction?
      impact_ratios.append(
        #math.log((i1s + i2s + 2) / total_points_clustered)
        ((i1s + i2s + 2) / total_points_clustered)
      * (absorbed / max(dominant, MIN_SIG_IMPACT/IMPACT_SIG_SIZE))
      )

      # Compute cluster size statistics:
      cluster_sizes = [i["size"] for i in clinfo.values()]
      nt_cluster_sizes = list(filter(lambda x: x >= MIN_SIZE, cluster_sizes))
      n_sig_clusters = len(nt_cluster_sizes)
      if n_sig_clusters == 0:
        n_sig_clusters = 1
      average_cluster_size.append(
        (
          sum(cluster_sizes) / len(cluster_sizes),
          sum(nt_cluster_sizes) / n_sig_clusters
        )
      )
      number_of_clusters.append((len(cluster_sizes), len(nt_cluster_sizes)))

      # Compute neighbor count & neighborhood comparison:
      nbc = 0
      for nb in nbi[fr]:
        if r2 == u.find(nb):
          nbc += 1
      for nb in nbi[to]:
        if r1 == u.find(nb):
          nbc += 1
      neighbor_counts.append(nbc)

      # Compute growth rate statistics:
      growth_rate.append(d - prev_d)
      growths = [d - i1["mean"], d - i2["mean"]]
      mgr = min(growths)
      xgr = max(growths)
      nm = ni["mean"]
      if nm == 0:
        nm = 1
      local_growth_rate.append(
        (
          min(i1s, i2s),
          mgr / nm,
          xgr / nm,
          (d - nm) / nm,
          abs(i1["mean"] - i2["mean"]) / nm,
          (mgr + xgr) / nm
        )
      )

      # Local structure:
      local_structures.append(
        (
          d / np.mean(nbd[fr]) + d / np.mean(nbd[to]),
          (
            max(i1s, 1) * (d / i1["mean"] if i1["mean"] > 0 else 1)
          + max(i2s, 1) * (d / i2["mean"] if i2["mean"] > 0 else 1)
          ) / max(i1s + i2s, 2),
          d / nm
        )
      )

      # Track normalized distances:
      normalized_distances.append(nr)

      # Track internal distances:
      nq = (
        ni["internal"]
      / (ni["mean"] * ((ni["size"] + 1) * ni["size"]) / 2)
      )
      if i1["mean"] and i1["size"] > 0:
        q1 = (
          i1["internal"]
        / (i1["mean"] * ((i1["size"] + 1) * i1["size"]) / 2)
        )
      else:
        q1 = 1
      if i2["mean"] and i2["size"] > 0:
        q2 = (
          i2["internal"]
        / (i2["mean"] * ((i2["size"] + 1) * i2["size"]) / 2)
        )
      else:
        q2 = 1

      internal_distances.append(
        (
          ni["internal"],
          ni["internal"] - max(i1["internal"], i2["internal"]),
          1 / nq,
          (1 / nq - 1 / max(q1, q2)),
        )
      )

      # Compute local linearity:
      lg1 = i1["largest"]
      lg2 = i2["largest"]
      # TODO: MATH
      predictions = []
      measures = []
      for lg in [lg1, lg2]:
        slope, intercept, r_val, p_val, stderr = linregress(range(len(lg)), lg)
        predictions.append(intercept + len(lg) * slope)
        slope, intercept, r_val, p_val, stderr = linregress(
          range(len(lg)+1),
          lg + [d]
        )
        #measures.append(slope)
        if len(lg) >= 2:
          measures.append((d - lg[-1]) - (lg[-1] - lg[-2]))
        else:
          measures.append(0)
        # TODO: This simpler method?
        #avdev = sum([lg[i] - lg[i-1] for i in range(1,len(lg))]) / (len(lg) - 1)
        #predictions.append(lg[-1] + avdev)
      # TODO: Which method here?
      #cd = max(d - p for p in predictions)
      ll = sum(measures) / len(measures)
      cd = sum(d - p for p in predictions) / len(predictions)
      lg = ni["largest"]
      mlg = np.mean(lg)
      if ni["size"] < MIN_SIZE:
        local_linearity.append((0, d))
      else:
        #local_linearity.append(cd / mlg)
        local_linearity.append([ll, d])

      # Compute size difference
      size_diff.append(abs(i1s - i2s))

      # Compute coherence
      coherence.append(
        (
          ni["coherence"],
          ni["coherence"] - max(i1["coherence"], i2["coherence"]),
          ni["coherence"] / ni["size"],
          ni["coherence"] / ni["size"] - max(
            i1["coherence"] / (i1s if i1s > 0 else 1),
            i2["coherence"] / (i2s if i2s > 0 else 1)
          ),
        )
      )

      # Compute std difference
      std_diff.append(
        ni["variance"]**0.5
      - (i1["variance"]**0.5 + i2["variance"]**0.5)/2
      )

      # Update previous distance
      prev_d = d

  print("  ...done.")

  # Plot averages:
  if "averages" in PLOT:
    acs = np.asarray(average_cluster_size)
    newfig()
    for i in range(1,acs.shape[1]):
      plot_data(acs[:,i], label=["all", "non-trivial"][i])
    plt.legend()
    plt.title("Average Cluster Size")

    nc = np.asarray(number_of_clusters)
    newfig()
    for i in range(1,nc.shape[1]):
      plot_data(nc[:,i], label=["all", "non-trivial"][i])
    plt.legend()
    plt.title("Number of Clusters")

    plt.figure()
    plt.semilogy(acs[:,1] / nc[:,1])
    plt.title("Average Cluster Size over Number of Clusters")

  # Plot std changes:
  sd = np.asarray(std_diff)
  if "std_diff" in PLOT:
    newfig()
    plot_data(sd, stats=True)
    plt.title("Change in Cluster Standard Deviation")

  lcv = np.asarray(local_linearity)
  if "local_linearity" in PLOT:
    newfig()
    plot_data(lcv[:,0])
    plot_data(lcv[:,1])
    plt.title("Local Linearity")

  idst = np.asarray(internal_distances)
  if "internal_distances" in PLOT:
    newfig()
    #plot_data(idst[:,0])
    #plot_data(idst[:,1])
    plot_data(idst[:,2])
    plt.axhline(0.5, color=(0.4, 0.4, 0.4))
    plot_data(idst[:,3], stats=True)
    plt.title("Internal Distance")

  nbc = np.asarray(neighbor_counts)
  if "neighbor_counts" in PLOT:
    newfig()
    plot_data(nbc, stats=True)
    plt.title("Combined Shared Neighbor Count")

  ipct = np.asarray(impacts)
  tpc = np.asarray(total_sizes)
  if "impact" in PLOT:
    newfig()
    #plot_data(tpc * IMPACT_SIG_SIZE, desat=True)
    plt.axhline(IMPACT_SIG_SIZE, color=DESAT[0])
    plot_data(ipct, smooth=OUTLIER_CRITERION)
    plt.title("Impact")

  ipctr = np.asarray(impact_ratios)
  if "impact_ratio" in PLOT:
    newfig()
    #plot_data(tpc * IMPACT_SIG_SIZE, desat=True)
    plt.axhline(IMPACT_SIG_SIZE, color=DESAT[0])
    plot_data(ipctr)
    plt.title("Impact Ratio")

  # Convert stuff to arrays:
  lgr = np.asarray(local_growth_rate)
  gra = np.asarray(growth_rate)
  nds = np.asarray(normalized_distances)
  n_edges = len(included)

  lstr = np.asarray(local_structures)
  lcmb = lstr[:,1]

  lstr_hist = np.convolve(lcmb, PREV_KERNEL, mode="same")
  lstr_hist = np.where(lstr_hist >= 1, lstr_hist, 1)

  # Construct a difference-space
  jd = lgr[:,3]
  mnm = lgr[:,4]
  dfs = np.concatenate(
    [ # normalize here before using a Euclidean metric:
      (mnm / np.median(mnm)).reshape(n_edges, 1),
      (jd / np.median(jd)).reshape(n_edges, 1),
      #(gra / np.median(gra)).reshape(n_edges, 1),
    ],
    axis=1
  )
    
  # Find "cut" edges:
  # Mixed hard criterion method:
  #b3 = 1.1 * np.convolve(lgr[:,3], KERNEL, mode="same")
  #b4 = 1.1 * np.convolve(lgr[:,4], KERNEL, mode="same")
  #cut = (lgr[:,3] > b3) * (lgr[:,4] > b4)

  # min+max critical threshold method:
  #mnm_ceil = np.median(mnm[:5])*1.02
  #cut = mnm > mnm_ceil

  # difference space outliers method:
  #dfs_distances = pairwise.pairwise_distances(dfs, metric="euclidean")
  #dfs_distances.sort(axis=1)
  ## TODO: use nearest or second-nearest here?
  ##dfs_distances = dfs_distances[:,1]
  #dfs_distances = dfs_distances[:,2]
  #dstd = np.std(dfs_distances)
  #dmean = np.mean(dfs_distances)
  #
  #cut = dfs_distances > (dmean + dstd*OUTLIER_CRITERION)

  # size growth outliers method:
  #sg = lgr[:,0]
  #mean_sg = np.mean(sg)
  #std_sg = np.std(sg)
  #cut = sg > mean_sg + std_sg * OUTLIER_CRITERION

  # combined local structure method:
  #cut = lcmb > lstr_hist * OUTLIER_CRITERION

  # impact ratio method:
  #cut = ipctr > IMPACT_SIG_SIZE

  # stdev change outliers method:
  #mean_sd = np.mean(sd)
  #std_sd = np.std(sd)
  #cut = sd > mean_sd + std_sd * OUTLIER_CRITERION

  # combined method:
  impact = ipctr / IMPACT_SIG_SIZE
  local = lcmb / (lstr_hist * OUTLIER_CRITERION)
  significance = (0.5 * impact + 0.5 * local) / SIG_THRESHOLD
  #cut = significance > 1.0

  # quality method
  cut = (idst[:,2] < 0.5) & (idst[:,3] < 0)

  colors = []
  for i in range(len(lgr)):
    if cut[i]:
      colors.append(COLORS[1])
    else:
      colors.append(COLORS[0])

  # Find clusterings according to cut edges:
  clusterings = [(uf.unionfind(n), [], 0.0)]
  for i, (fr, to, d, nr) in enumerate(reversed(included)):
    ri = len(included) - i - 1
    for clstr in clusterings:
      clstr[0].unite(fr, to)
      clstr[1].append((fr, to, d, nr))
    # TODO: Formalize momentum?
    #if cut[ri] and (ri == 0 or not cut[ri-1]):
    if cut[ri]:
      clusterings.append((uf.unionfind(n), [], significance[ri]))

  if "significance" in PLOT:
    newfig()
    plt.axhline(1.0, color=DESAT[0])
    plot_data(significance)
    plt.title("Significance")

  if "distance_space" in PLOT:
    plt.figure()
    indexed = np.asarray(list(zip(range(len(dfs_distances)), dfs_distances)))
    plt.scatter(indexed[:,0], indexed[:,1], s=1, color=colors)
    plt.axhline(dmean, color=(0, 0, 0), linewidth=0.2)
    plt.axhline(
      dmean + dstd * OUTLIER_CRITERION,
      color=(0, 0, 0),
      linewidth=0.2
    )

  coh = np.asarray(coherence)
  if "coherence" in PLOT:
    newfig()
    for i in range(coh.shape[1]):
      plot_data(
        coh[:,i],
        label=["raw", "delta", "norm", "norm-delta"][i],
        smooth=OUTLIER_CRITERION,
        baseline=1.02
      )
      plt.plot(smooth, color=dc)
    plt.legend()
    plt.title("Coherence")

  if "local" in PLOT:
    newfig()
    plt.axhline(0, color=(0, 0, 0))
    for i in range(lgr.shape[1]):
      plot_data(
        lgr[:,i],
        label=["size", "min", "max", "avg", "join", "min+max"][i],
        smooth=OUTLIER_CRITERION,
        baseline=1.02
      )
    plt.legend()
    plt.title("Local Growth Rate")

  if "local_structure" in PLOT:
    newfig()
    for i in range(lstr.shape[1]):
      plot_data(
        lstr[:,i],
        smooth=OUTLIER_CRITERION,
        label=["neighborhood", "combined", "average"][i]
      )
    plt.legend()
    plt.title("Local Structure")

  if "normalized_distances" in PLOT:
    newfig()
    plt.axhline(1, color=(0.5, 0.5, 0.5))
    plot_data(nds, smooth=OUTLIER_CRITERION)
    plt.title("Normalized Distances")

  if "lcompare" in PLOT:
    colors = []
    for i in range(lgr.shape[0]):
      if cut[i]:
        colors.append(COLORS[1])
      else:
        colors.append(COLORS[0])

    axl = ["Edge Difference", "Join Difference", "Raw Difference"]
    if dfs.shape[1] == 2:
      plt.figure()
      plt.scatter(dfs[:,0], dfs[:,1], s=1, color=colors)
      plt.xlabel(axl[0])
      plt.ylabel(axl[1])
      plt.title("Difference Space")
    if dfs.shape[1] == 3:
      fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
      for i, projection in enumerate([(0, 1), (1, 2), (0, 2)]):
        ax = axes[i]
        x, y = projection
        ax.scatter(dfs[:,x], dfs[:,y], s=1, color=colors)
        ax.set_xlabel(axl[x])
        ax.set_ylabel(axl[y])
        ax.set_title("Difference Space")

  # Plot edges with cuts highlighted:
  if "cut" in PLOT:
    ep = []
    ec = []
    climit = max(e[2] for e in included) * 1.5
    for i, (fr, to, d, nr) in enumerate(included):
      ep.append([projected[fr], projected[to]])
      if cut[i]:
        ec.append([1, 0, 0])
      else:
        ec.append([d / climit]*3)

    lc = mc.LineCollection(ep, colors=ec, linewidths=0.8)
    fig, ax = plt.subplots()
    plt.scatter(projected[:,0], projected[:,1], s=1.2, c=POINT_COLOR)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

  # Plot edge quartiles:
  if "quartiles" in PLOT:
    regions = 4
    ep = []
    ec = []
    for i, (fr, to, d, nr) in enumerate(included):
      percentile = int(regions * (i / len(included)))
      ep.append([projected[fr], projected[to]])
      ec.append(COLORS[percentile % len(COLORS)])
    fig, ax = plt.subplots()
    lc = mc.LineCollection(ep, colors=ec, linewidths=0.8)
    plt.scatter(projected[:,0], projected[:,1], s=1.2, c=POINT_COLOR)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

  # Plot projected points
  # TODO: Plot multiple projections?
  if "raw" in PLOT:
    plt.figure()
    plt.scatter(projected[:,0], projected[:,1], s=0.5, c=POINT_COLOR)

  # Plot absolute growth rate:
  if "absolute_growth" in PLOT:
    mean = np.mean(gra)

    std = np.std(gra)
    plt.figure()
    plt.plot(gra)
    plt.axhline(mean, color=(1.0, 0, 0))
    plt.axhline(mean + std, color=(1.0, 0.5, 0))
    plt.axhline(mean - std, color=(1.0, 0.5, 0))
    plt.xlabel("Absolute Growth Rate")

  # Plot edge lengths:
  if "edge_lengths" in PLOT:
    plt.figure()
    plt.plot([e[2] for e in sorted_edges])
    plt.xlabel("Edge Lengths")

  # Plot included edge lengths:
  if "included_lengths" in PLOT:
    plt.figure()
    clst = sorted(clusterings, key=lambda x: x[2])[-1] # most-significant
    #clst = clusterings[:2][-1] # second-to-last clustering if there is one
    unions, edges, sig = clst
    splits = {}
    for fr, to, d, nr in edges:
      i = unions.find(fr)
      if i not in splits:
        splits[i] = []
      splits[i].append(d)

    for i, sp in enumerate(splits.values()):
      plt.scatter(
        np.arange(len(sp)),
        sp,
        color= COLORS[i % len(COLORS)],
        s=0.75
      )
    plt.xlabel("Included Edge Lengths")

  # Plot clustering results:
  if "results" in PLOT:

    sqw = int(math.ceil(len(clusterings)**0.5))
    sqh = sqw
    while len(clusterings) <= sqw * sqh - sqw:
      sqh -= 1
    if sqw * sqh == 1:
      fix, ax = plt.subplots()
      plt.title("Singular Clustering")
      plt.axis("equal")
      plt.scatter(projected[:,0], projected[:,1], s=0.8, c=POINT_COLOR)
      data = []
      colors = []
      unions, edges, sig = clusterings[0]
      for fr, to, d, nr in edges:
        colors.append(COLORS[unions.find(fr) % len(COLORS)])
        data.append((projected[fr], projected[to]))
      data = np.asarray(data)
      lc = mc.LineCollection(
        data,
        colors=colors,
        linewidths=0.8
      )
      ax.add_collection(lc)
      ax.autoscale()
      ax.margins(0.1)
    else:
      fig, axes = plt.subplots(sqh, sqw, sharex=True, sharey=True)
      plt.axis("equal")
      fig.suptitle("{} Clusterings".format(len(clusterings)))
      for i, clstr in enumerate(clusterings):
        sqx, sqy = i % sqw, i // sqw
        data = []
        colors = []
        unions, edges, sig = clstr
        for fr, to, d, nr in edges:
          colors.append(COLORS[unions.find(fr) % len(COLORS)])
          data.append((projected[fr], projected[to]))
        data = np.asarray(data)
        lc = mc.LineCollection(
          data,
          colors=colors,
          linewidths=0.8
        )
        if sqh == 1:
          axes[sqx].scatter(
            projected[:,0],
            projected[:,1],
            s=1.2,
            c=POINT_COLOR
          )
          axes[sqx].add_collection(lc)
          axes[sqx].autoscale()
          axes[sqx].margins(0.1)
        else:
          axes[sqy][sqx].scatter(
            projected[:,0],
            projected[:,1],
            s=1.2,
            c=POINT_COLOR
          )
          axes[sqy][sqx].add_collection(lc)
          axes[sqy][sqx].autoscale()
          axes[sqy][sqx].margins(0.1)

  if "result_stats" in PLOT:
    scales = []
    coverages = []
    between = []
    within = []
    qualities = []
    significances = []
    for clstr in clusterings:
      unions, edges, sig = clstr
      significances.append(sig)
      groups = unions.groups()

      realgroups = [g for g in groups if len(g) >= MIN_SIZE]
      realcount = len(realgroups)

      if realcount == 0:
        continue

      da = np.asarray(edges)[:,2]
      scales.append(np.mean(da))

      clustered_count = 0
      tbcd = 0
      awcd = 0
      nstds = []
      for i, g in enumerate(realgroups):
        rep = g[0]
        clustered_count += len(g)

        ehere = []
        norms = []
        wcd = 0
        for fr, to, d, nr in edges:
          if unions.issame(fr, rep):
            ehere.append((fr, to, d, nr))
            norms.append(nr)
            wcd += d
        wcd /= len(g)
        awcd += wcd
        nstds.append(np.std(norms))

        bcds = []
        for og in realgroups[i:]:
          if g != og and len(og) >= MIN_SIZE:
            bcds.append(
              min(
                euclidean(points[p1], points[p2])
                  for p1 in g
                  for p2 in og
              )
            )
        tbcd += min(bcds) if bcds else 0

      coverages.append(clustered_count / n)

      abcd = tbcd / realcount
      awcd /= realcount

      between.append(abcd)
      within.append(awcd)

      qualities.append(1 - np.mean(nstds))
      
    scales = np.asarray(list(reversed(scales)))
    scales /= np.max(scales)
    coverages = list(reversed(coverages))
    between = list(reversed(between))
    within = list(reversed(within))
    qualities = list(reversed(qualities))
    significances = np.asarray(list(reversed(significances)))
    significances /= np.max(significances)

    plt.figure()
    plt.plot(scales, label="scale")
    plt.plot(coverages, label="coverage")
    plt.plot(between, label="between")
    plt.plot(within, label="within")
    plt.plot(qualities, label="quality")
    plt.plot(significances, label="significance")
    plt.legend()
    plt.xlabel("Cluster Stats")

  #plt.show()
  print("...done with clustering.")

  best = sorted(clusterings, key=lambda x: x[2])[-1]
  unions, edges, sig = best
  raw_asg = list(set([unions.find(i) for i in range(len(points))]))
  assignments = [raw_asg.index(unions.find(i)) for i in range(len(points))]

  return points, (nbd, nbi), best, assignments, clusterings

def subclusters(points, neighbors, clustering):
  nbd, nbi = neighbors
  unions, edges, sig = clustering
  analyze = []
  for g in unions.groups():
    if len(g) >= ANALYSIS_SIZE:
      analyze.append(g)

  results = []
  for g in analyze:
    sc = {
      "points": [],
      "core_edges": [],
      "all_edges": [],
    }
    for pid in g:
      sc["points"].append(pid)
      for i, dst in enumerate(nbi[pid]):
        if dst in g:
          sc["all_edges"].append((pid, dst, nbd[pid,i]))

    anchor = unions.find(g[0])
    for fr, to, d, nr in edges:
      if unions.find(fr) == anchor:
        sc["core_edges"].append((fr, to, d))
        # ensure connectivity of all_edges graph:
        if (fr, to, d) not in sc["all_edges"]:
          sc["all_edges"].append((fr, to, d))

    results.append(sc)

  return results

def analyze_clustering(points, neighbors, clustering):
  subcl = subclusters(points, neighbors, clustering)
  if len(subcl) == 0:
    print("No interesting subclusters to analyze.")
  else:
    for cluster in subcl:
      manifold = approximate_manifold(points, cluster)
      mdim = 1 + (len(manifold["anchors"]) - 2) / 2
      print("Cluster size: {}".format(len(cluster["points"])))
      print("Manifold edges: {}".format(len(cluster["all_edges"])))
      print("Manifold dimensions: {}".format(mdim))
      if "manifolds" in PLOT:
        plot_manifold(manifold, show_interior=False, show_edges=True)
    #plt.show()

def plot_manifold(
  manifold,
  show_interior=False,
  show_grid=False,
  show_edges=False
):
    n = manifold["exterior"].shape[0]
    nanchors = len(manifold["anchors"])
    mdim = 1 + (nanchors - 2) / 2
    fig, ax = plt.subplots()
    plt.title("Detected Manifold ({}-dimensional)".format(mdim))
    plt.axis("equal")

    # Manifold edges:
    earray = []
    for fr, to, d in manifold["edges"]:
      earray.append((manifold["exterior"][fr], manifold["exterior"][to]))
    earray = np.asarray(earray)
    lc = mc.LineCollection(
      earray,
      colors=(0.8, 0.8, 0.8),
      linewidths=0.7
    )
    ax.add_collection(lc)

    # Original points:
    plt.scatter(
      manifold["exterior"][:,0],
      manifold["exterior"][:,1],
      s=1.0,
      c=[(0.9, 0.9, 0.9)]*n
    )

    # Anchor points:
    apoints = manifold["exterior"][manifold["anchors"],:]
    plt.scatter(
      apoints[:,0],
      apoints[:,1],
      s=5.0,
      c=[(1.0, 0.0, 0.0)]*apoints.shape[0]
    )

    # Reconstructed points:
    plt.scatter(
      manifold["reconstructed"][:,0],
      manifold["reconstructed"][:,1],
      s=0.8,
      c=[(0.0, 0.4, 0.05)]*n
    )

    # Interior Points (at most 2 anchors-worth):
    if show_interior:
      if mdim < 1:
        plt.scatter(
          manifold["interior"][:,0],
          np.full(
            (manifold["interior"].shape[0],),
            np.mean(manifold["exterior"][:1])
          ),
          s=0.8,
          c=[(0.0, 0.1, 0.6)]*n
        )
      else:
        plt.scatter(
          manifold["interior"][:,0],
          manifold["interior"][:,1],
          s=0.8,
          c=[(0.4, 0.6, 1.0)]*n
        )

    # Grid:
    if show_grid:
      print("Building manifold grid...")
      # A uniform grid over a unit simplex:
      gridpoints = simplex_grid(
        nanchors,
        MANIFOLD_GRID_RESOLUTION
      ) / MANIFOLD_GRID_RESOLUTION
      neighbors = NearestNeighbors(
        n_neighbors=nanchors,
        algorithm="ball_tree"
      ).fit(gridpoints)
      nbd, nbi = neighbors.kneighbors(gridpoints)
      gridedges = []
      for i in range(len(nbi)):
        for nb in nbi[i]:
          gridedges.append((i, nb))
      print("  ...done building grid.")

      print(
        "Translating grid into original space ({} points)...".format(
          len(gridpoints)
        )
      )
      extpoints = []
      for i, gp in enumerate(gridpoints):
        prbar(i / len(gridpoints))
        extpoints.append(approximate_exterior_position(manifold, gp))
      extpoints = np.asarray(extpoints)

      extedges = [ (extpoints[i], extpoints[j]) for (i, j) in gridedges ]
      print("  ...done translating grid.")

      plt.scatter(
        extpoints[:,0],
        extpoints[:,1],
        s=0.4,
        c=[(0.0, 0.0, 0.0)]*n
      )
      lc = mc.LineCollection(
        extedges,
        colors=(0.2, 0.2, 0.2),
        linewidths=0.7
      )
      ax.add_collection(lc)

    # Edges:
    if show_edges:
      print("Building manifold edges...")
      extrema = np.asarray([
        [0]*i + [1] + [0]*(nanchors - i - 1)
          for i in range(nanchors)
      ])
      edges = []
      for i, fr in enumerate(extrema):
        for j, to in enumerate(extrema[i:]):
          edges.append(np.asarray([
            fr*(interp/MANIFOLD_EDGE_SAMPLES)
          + to * (MANIFOLD_EDGE_SAMPLES - interp)/MANIFOLD_EDGE_SAMPLES
              for interp in range(MANIFOLD_EDGE_SAMPLES+1)
          ]))
      print("  ...done building edges.")


      print("Translating edges into original space...")
      exedges = []
      for e in edges:
        ex = []
        for p in e:
          ex.append(approximate_exterior_position(manifold, p))
        exedges.append(ex)
      exedges = np.asarray(exedges)

      elen = exedges.shape[1]

      exlines = []
      for e in exedges:
        el = []
        for i in range(len(e)-1):
          el.append((e[i], e[i+1]))
        exlines.append(el)
      exlines = np.asarray(exlines)
      print("  ...done translating edges.")

      for i in range(len(exedges)):
        plt.scatter(
          exedges[i][:,0],
          exedges[i][:,1],
          s=0.4,
          c=[(0.0, 0.0, 0.0)]*elen
        )
        lc = mc.LineCollection(
          exlines[i],
          colors=[(0.2, 0.2, 0.2)]*(elen-1),
          linewidths=0.7
        )
        ax.add_collection(lc)

def floyd_warshall(earray):
  # TODO: Get rid of this?
  n = earray.shape[0]
  for shorter in range(n):
    for fr in range(n):
      #if fr == shorter:
      #  continue
      for to in range(n):
        #if fr == to:
        #  continue
        better = earray[fr,shorter] + earray[shorter, to]
        if better < earray[fr, to]:
          earray[fr, to] = better

def collapse_cluster(points, cluster):
  result = {
    "count": 0,
    "scale": 0,
    "points": [],
    "edges": [],
  }
  pidmap = {}

  for pid in cluster["points"]:
    pidmap[pid] = len(result["points"])
    result["points"].append(points[pid])

  result["points"] = np.asarray(result["points"])
  result["count"] = result["points"].shape[0]

  result["scale"] = 0
  result["edges"] = []
  for fr, to, d in cluster["all_edges"]:
    result["scale"] += d
    result["edges"].append((pidmap[fr], pidmap[to], d))

  result["scale"] /= len(result["edges"])

  return result

# TODO: Relax points beforehand? Deal with noise better somehow!
def approximate_manifold(points, cluster):
  # Find the shortest-path-distances-matrix for our cluster:
  local = collapse_cluster(points, cluster)

  longest_edge = max(d for fr, to, d in local["edges"])
  nonedge = longest_edge*local["count"]**2
  edgematrix = np.full((local["count"], local["count"]), nonedge)
  for fr, to, d in local["edges"]:
    edgematrix[fr,to] = d
    edgematrix[to,fr] = d

  for i in range(local["count"]):
    edgematrix[i,i] = 0

  shortest_paths = graph_shortest_paths(edgematrix)
  if np.max(shortest_paths) == nonedge:
    raise RuntimeError("Cluster edges graph is not fully connected!")

  start = 0 # pick an arbitrary point
  extrema = [np.argmax(shortest_paths[start,:])]
  extrema.append(np.argmax(shortest_paths[extrema[0],:]))
  manifold_approximation = {
    "count": local["count"],
    "scale": local["scale"],
    "anchors": extrema,
    "exterior": local["points"],
    "interior": [],
    "reconstructed": [],
    "edges": local["edges"],
    "paths": shortest_paths,
  }
  compute_interior_positions(manifold_approximation)
  rerr = compute_reconstruction(manifold_approximation)
  prerr = rerr
  improvement = 1.0
  icount = 0
  print(
    "Baseline reconstruction error ({} anchors): {:.5f}".format(
      len(manifold_approximation["anchors"]),
      rerr
    )
  )
  out_of_extrema = False
  while improvement > MANIFOLD_ERROR_THRESHOLD:
    icount += 1
    distances = manifold_approximation["paths"][
      manifold_approximation["anchors"],
      :
    ]
    combined = np.sum(distances**0.5, axis=0)**2
    bidx = np.argmax(combined)

    if bidx in manifold_approximation["anchors"]:
      print(
        "Warning: failed to find new extremum (found {} given {}).".format(
          bidx,
          manifold_approximation["anchors"],
        )
      )
      out_of_extrema = True
      break
    manifold_approximation["anchors"].append(bidx)
    compute_interior_positions(manifold_approximation)
    rerr = compute_reconstruction(manifold_approximation)
    improvement = prerr - rerr
    prerr = rerr
    print(
      "[{}] Median reconstruction error ({} anchors): {:.5f} {:.3f}".format(
        icount,
        len(manifold_approximation["anchors"]),
        rerr,
        -improvement
      )
    )

  # Last anchor didn't improve things much, so remove it (not the case if we
  # ran out of extrema).
  if not out_of_extrema:
    manifold_approximation["anchors"] = manifold_approximation["anchors"][:-1]
    compute_interior_positions(manifold_approximation)
    compute_reconstruction(manifold_approximation)

  return manifold_approximation

def find_manifold_position(manifold, manifold_point_index):
  alines = manifold["paths"][manifold_point_index, manifold["anchors"]]
  return alines / np.sum(alines)

def compute_interior_positions(manifold):
  # Compute interior points
  manifold["interior"] = []
  for mpi in range(manifold["count"]):
    manifold["interior"].append(find_manifold_position(manifold, mpi))
  manifold["interior"] = np.asarray(manifold["interior"])
  # Compute interior scale
  iedges = []
  for (fr, to, d) in manifold["edges"]:
    iedges.append(euclidean(manifold["interior"][fr], manifold["interior"][to]))
  manifold["interior_scale"] = np.percentile(
    iedges,
    MANIFOLD_INTERIOR_SCALE_PERCENTILE
  )

def approximate_exterior_position(manifold, manifold_position):
  best = []
  for i in range(manifold["count"]):
    d = euclidean(manifold["interior"][i], manifold_position)
    best.append([i, d])

  # Sort by distance
  apx_scale = manifold["interior_scale"] * MANIFOLD_APPROX_SCALE_MULTIPLIER
  best = sorted(best, key=lambda x: x[1])
  chosen = [best[0]]
  best = best[1:]
  while (
    best
and (
      len(chosen) < MANIFOLD_MIN_NEIGHBORHOOD
   or chosen[-1][1] < apx_scale
    )
  ):
    chosen.append(best.pop(0))

  # expand scale if necessary
  max_dist = chosen[-1][1]
  if max_dist > apx_scale:
    apx_scale = max_dist * 1.1

  # compute strengths:
  strsum = 0
  for i in range(len(chosen)):
    bi, bd = chosen[i]
    strength = math.cos((bd / apx_scale) * (math.pi/2))
    strsum += strength
    chosen[i].append(strength)

  # normalize strengths into weights:
  for i in range(len(chosen)):
    chosen[i][2] /= strsum

  # compute approximation via averaging:
  approx = np.zeros((manifold["exterior"].shape[1],))
  for bi, bd, bw in chosen:
    approx += manifold["exterior"][bi] * bw
  return approx

def compute_reconstruction(manifold):
  errors = []
  manifold["reconstructed"] = []
  for i, ip in enumerate(manifold["interior"]):
    rp = approximate_exterior_position(manifold, ip)
    manifold["reconstructed"].append(rp)
    errors.append(euclidean(rp, manifold["exterior"][i]) / manifold["scale"])

  manifold["reconstructed"] = np.asarray(manifold["reconstructed"])

  return np.median(errors)


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
    points, neighbors, best, assignments, clusterings = cluster(tc)
    analyze_clustering(points, neighbors, best)
    plt.show()
  #cluster(IRIS_DATA)

if __name__ == "__main__":
  test()

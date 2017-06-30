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
from scipy.stats import linregress

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

N_LARGEST = 5

OUTLIER_CRITERION = 1.0

PREVENT_ZERO_MEANS = False
EPSILON = 0.00000000001

POINT_COLOR = (0, 0, 0)

# Whether or not to normalize distances
NORMALIZE_DISTANCES = True
NORMALIZATION_NEIGHBORHOOD = 7
NORMALIZATION_STRENGTH = 0.05

PLOT = [
  #"averages",
  #"distance_space",
  #"std_diff",
  #"local_linearity",
  "neighbor_counts",
  #"local",
  #"lcompare",
  #"cut",
  #"quartiles",
  #"edge_lengths",
  "included_lengths",
  #"raw",
  #"absolute_growth",
  "results",
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
      gring((0.2, 0.6), (0.05, 0.05), 0.01, 100),
      gring((0.7, 0.3), (0.07, 0.07), 0.01, 100),
    ]
  ),
  np.concatenate(
    [
      gring((0.5, 0.5), (0.5, 0.5), 0.04, 100),
      gring((0.5, 0.5), (0.2, 0.2), 0.08, 100),
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

def combine_info(A, B):
  sA = A["size"]
  sB = B["size"]
  mA = A["mean"]
  mB = B["mean"]
  vA = A["variance"]
  vB = B["variance"]

  if sA == 0 or sB == 0:
    return {
      "size": sA + sB,
      "vertices": A["vertices"] | B["vertices"],
      "edges": A["edges"] | B["edges"],
      "mean": max(A["mean"], B["mean"]),
      "variance": max(A["variance"], B["variance"]),
      "largest": sorted(A["largest"] + B["largest"])[-N_LARGEST:],
    }

  result = {}
  result["size"] = sA + sB
  result["vertices"] = A["vertices"] | B["vertices"]
  result["edges"] = A["edges"] | B["edges"]
  result["largest"] = sorted(A["largest"] + B["largest"])[-N_LARGEST:]

  result["mean"] = (mA * sA + mB * sB) / (sA + sB)
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
  return result

def prbar(progress):
  pbwidth = 70
  sofar = int(pbwidth * progress)
  left = pbwidth - sofar - 1
  print("\r[" + "="*sofar + ">" + "-"*left + "]", end="")

def cluster(points, metric="euclidean"):
  print("Starting clustering process...")
  print("  ...computing pairwise distances...")
  distances = pairwise.pairwise_distances(points, metric=metric)
  if NORMALIZE_DISTANCES:
    nearest = np.sort(distances, axis=1)[:,1:NORMALIZATION_NEIGHBORHOOD+1]
    local_scale = np.median(nearest, axis=1)
    #distances = np.divide(distances, local_scale, axis=1)
    distances = (
      (distances / local_scale * NORMALIZATION_STRENGTH)
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
      edges.append((fr, to, distances[fr,to]))
      # TODO: non-symmetric metrics?
      #edges.append((to, fr, distances[to,fr]))

  sorted_edges = sorted(edges, key=lambda e: e[2])
  u = uf.unionfind(distances.shape[0])

  prev_d = 0
  growth_rate = []
  local_growth_rate = []
  local_linearity = []
  average_cluster_size = []
  number_of_clusters = []
  std_diff = []
  size_diff = []
  neighbor_counts = []
  included = []
  clinfo = {}
  for i in range(n):
    clinfo[i] = {
      "size": 0,
      "vertices": { i },
      "edges": set(),
      "mean": 0,
      "variance": 0,
      "largest": [0] * N_LARGEST,
    }
  print("  ...constructing MST...")
  for i, (fr, to, d) in enumerate(sorted_edges):
    prbar(i / len(sorted_edges))
    if not u.issame(fr, to):
      r1 = u.find(fr)
      r2 = u.find(to)
      i1 = clinfo[r1]
      i2 = clinfo[r2]
      i1s = i1["size"]
      i2s = i2["size"]
      ni = combine_info(i1, i2)
      ni = combine_info(
        ni,
        {
          "size": 1,
          "vertices": { fr, to },
          "edges": { (fr, to, d) },
          "mean": d,
          "variance": 0,
          "largest": [0] * N_LARGEST + [d],
        }
      )
      del clinfo[r1]
      del clinfo[r2]
      included.append((fr, to, d))
      u.unite(fr, to)
      clinfo[u.find(fr)] = ni

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

      # Compute neighbor count:
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

      # Compute local linearity:
      lg1 = i1["largest"]
      lg2 = i2["largest"]
      # TODO: MATH
      predictions = []
      for lg in [lg1, lg2]:
        slope, intercept, r_val, p_val, stderr = linregress(range(len(lg)), lg)
        predictions.append(intercept + len(lg) * slope)
        # TODO: This simpler method?
        #avdev = sum([lg[i] - lg[i-1] for i in range(1,len(lg))]) / (len(lg) - 1)
        #predictions.append(lg[-1] + avdev)
      # TODO: Which method here?
      #cd = max(d - p for p in predictions)
      cd = sum(d - p for p in predictions) / len(predictions)
      lg = ni["largest"]
      mlg = np.mean(lg)
      if ni["size"] < MIN_SIZE:
        local_linearity.append(0)
      else:
        local_linearity.append(cd / mlg)

      # Compute size difference
      size_diff.append(abs(i1s - i2s))

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
    plt.figure()
    for i in range(1,acs.shape[1]):
      c = COLORS[i % len(COLORS)]
      plt.plot(acs[:,i], color=c, label=["all", "non-trivial"][i])
      plt.legend()
      plt.title("Average Cluster Size")

    nc = np.asarray(number_of_clusters)
    plt.figure()
    for i in range(1,nc.shape[1]):
      c = COLORS[i % len(COLORS)]
      plt.plot(nc[:,i], color=c, label=["all", "non-trivial"][i])
      plt.legend()
      plt.title("Number of Clusters")

    plt.figure()
    plt.semilogy(acs[:,1] / nc[:,1])
    plt.title("Average Cluster Size over Number of Clusters")

  # Plot std changes:
  sd = np.asarray(std_diff)
  if "std_diff" in PLOT:
    plt.figure()
    plt.axhline(np.mean(sd), color=DESAT[0])
    plt.axhline(np.mean(sd) + np.std(sd), color=DESAT[0])
    plt.plot(sd, color=COLORS[0])
    plt.title("Change in Cluster Standard Deviation")

  lcv = np.asarray(local_linearity)
  if "local_linearity" in PLOT:
    plt.figure()
    plt.axhline(np.mean(lcv), color=DESAT[0])
    plt.axhline(np.mean(lcv) + np.std(lcv), color=DESAT[0])
    plt.plot(lcv, color=COLORS[0])
    plt.title("Local Curvature")

  nbc = np.asarray(neighbor_counts)
  if "neighbor_counts" in PLOT:
    plt.figure()
    plt.axhline(np.mean(nbc), color=DESAT[0])
    plt.axhline(np.mean(nbc) + np.std(nbc), color=DESAT[0])
    plt.plot(nbc, color=COLORS[0])
    plt.title("Combined Shared Neighbor Count")

  # Plot local growth rate:
  lgr = np.asarray(local_growth_rate)
  gra = np.asarray(growth_rate)
  n_edges = len(included)

  kernel = np.asarray([0.75, 0.75, 1, 1, 1, 1, 1, 0.75, 0.75])
  kernel /= sum(kernel)

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
  #b3 = 1.1 * np.convolve(lgr[:,3], kernel, mode="same")
  #b4 = 1.1 * np.convolve(lgr[:,4], kernel, mode="same")
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
  sg = lgr[:,0]
  mean_sg = np.mean(sg)
  std_sg = np.std(sg)
  cut = sg > mean_sg + std_sg * OUTLIER_CRITERION

  # stdev change outliers method:
  #mean_sd = np.mean(sd)
  #std_sd = np.std(sd)
  #cut = sd > mean_sd + std_sd * OUTLIER_CRITERION

  colors = []
  for i in range(len(lgr)):
    if cut[i]:
      colors.append(COLORS[1])
    else:
      colors.append(COLORS[0])

  # Find clusterings according to cut edges:
  clusterings = [(uf.unionfind(n), [])]
  for i, (fr, to, d) in enumerate(reversed(included)):
    ri = len(included) - i - 1
    for cl in clusterings:
      cl[0].unite(fr, to)
      cl[1].append((fr, to, d))
    # TODO: Formalize momentum?
    #if cut[ri] and (ri == 0 or not cut[ri-1]):
    if cut[ri]:
      clusterings.append((uf.unionfind(n), []))

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

  if "local" in PLOT:
    plt.figure()
    plt.axhline(0, color=(0, 0, 0))
    for i in range(lgr.shape[1]):
      this = lgr[:,i]
      ceil = np.median(this[:5])*1.02
      smooth = np.convolve(
        this,
        kernel,
        mode="same"
      )
      c = COLORS[i % len(COLORS)]
      dc = DESAT[i % len(DESAT)]
      plt.axhline(ceil, color=dc)
      plt.plot(
        this,
        color=c,
        label=["size", "min", "max", "avg", "join", "min+max"][i]
      )
      plt.plot(smooth, color=dc)
      plt.legend()
      plt.title("Local Growth Rate")

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
    for i, (fr, to, d) in enumerate(included):
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
    for i, (fr, to, d) in enumerate(included):
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
    clst = clusterings[:2][-1] # second-to-last clustering if there is one
    unions = clst[0]
    splits = {}
    for fr, to, d in clst[1]:
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
      plt.scatter(projected[:,0], projected[:,1], s=1.2, c=POINT_COLOR)
      data = []
      for fr, to, d in clusterings[0][1]:
        data.append(projected[fr], projected[to])
      data = np.asarray(data)
      lc = mc.LineCollection(
        data,
        colors=COLORS[i % len(COLORS)],
        linewidths=0.8
      )
      ax.add_collection(lc)
      ax.autoscale()
      ax.margins(0.1)
    else:
      fig, axes = plt.subplots(sqh, sqw, sharex=True, sharey=True)
      fig.suptitle("{} Clusterings".format(len(clusterings)))
      for i, cl in enumerate(clusterings):
        sqx, sqy = i % sqw, i // sqw
        data = []
        for fr, to, d in cl[1]:
          data.append((projected[fr], projected[to]))
        data = np.asarray(data)
        lc = mc.LineCollection(
          data,
          colors=COLORS[i % len(COLORS)],
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

  plt.show()
  print("...done with clustering.")

  return u.groups()

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
  for tc in test_cases[6:]:
  #for tc in test_cases:
    cluster(tc)
  #cluster(IRIS_DATA)

if __name__ == "__main__":
  test()

"""
multiscale.py
Implements a multi-scale minimum-spanning-tree-based clustering algorithm given
a distance matrix.
"""

# local imports:
import utils

import sys
import math
import multiprocessing
import copy

import unionfind as uf

import numpy as np

from scipy.stats import linregress
from scipy.spatial.distance import euclidean
from scipy.optimize import curve_fit

from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path

DEFAULT_CLUSTERING_PARAMETERS = {
  "quiet": True,
  # Distance parameters:
  "distances": None,
  "edges": None,
  #"metric": euclidean,
  "metric": "euclidean",
  "normalize_distances": False,
  "neighborhood_size": 10,
  "neighbors_cache_name": "multiscale-neighbors",
  "cache_neighbors": True,
  # Cluster detection parameters:
  "join_quality_threshold": 0.98,
  "absolute_size_threshold": 6,
  "relative_size_threshold": 0.005,
  "scale_increase_threshold": 1.1,
  #"remember": "size",
  #"remember": "quality",
  #"remember": "size_and_scale",
  "remember": None,
  # Separated parameters
  "interesting_size": 10,
  "condensation_threshold": 1.0,
  "condensation_quality": "mixed_quality",
  "decanting_quality": "quality",
}

DEFAULT_TYPICALITY_PARAMETERS = {
  "quiet": True,
  "suppress_warnings": False,
  "neighbors": None,
  "metric": "euclidean",
  "significant_fraction": 0.01,
  "min_neighborhood_size": 15,
  "parallelism": 16,
}

NEXT_CLUSTER_ID = 0

def join_clusters(A, B, e):
  """
  Combines two clusters being connected by an edge e, returning a new cluster
  representing the combination (the original clusters aren't modified). The
  quality of the new cluster is also returned, and is added to the given edge
  as a fourth element.

  It incrementally maintains the cluster mean edge length as well as the
  internal-measure for the cluster, which is the sum of the similarities
  between each pair of points in the cluster, where the similarity between two
  points is defined by the maximum length of any edge on the (shortest) path
  between them (of course, in a minimum spanning tree there's only ever a
  single path between points).

  Notably, when two clusters whose points are all close to each other are
  joined by an edge that's long relative to their existing edges, the
  internal-measure of the resulting cluster is much larger than those of the
  clusters being combined. This effect is less pronounced when the edge joining
  the clusters is roughly the same length as edges already in the clusters,
  especially if we control for cluster size (see the add_quality function
  below).
  """
  global NEXT_CLUSTER_ID

  fr, to, d = e
  sA = A["size"]
  sB = B["size"]
  mA = A["mean"]
  mB = B["mean"]

  combined_isolation = {}

  for v in A["isolation"]:
    avg, count = A["isolation"][v]
    combined_isolation[v] = (
      (avg * count + d * (sB + 1)) / (count + sB + 1),
      count + sB + 1
    )

  for v in B["isolation"]:
    avg, count = B["isolation"][v]
    combined_isolation[v] = (
      (avg * count + d * (sA + 1)) / (count + sA + 1),
      count + sA + 1
    )

  all_isolations = np.asarray([ic[0] for ic in combined_isolation.values()])
  isolation_mean = np.mean(all_isolations)
  isolation_std = np.std(all_isolations)

  outliers = {
    v
      for v in combined_isolation
      if combined_isolation[v][0] > isolation_mean + isolation_std
  }

  nc = {
    "id": NEXT_CLUSTER_ID,
    "size": sA + sB + 1,
    "scale": d,
    "parent": None,
    "children": [A, B],
    "vertices": A["vertices"] | B["vertices"],
    "isolation": combined_isolation,
    "isolation_mean": isolation_mean,
    "outliers": outliers,
    "core_size": sA + sB + 1 - len(outliers),
    "edges": A["edges"] | B["edges"], # new edge not included yet
    "mean": (mA * sA + mB * sB + d) / (sA + sB + 1),
    "internal": A["internal"] + B["internal"] + d * ((sA + 1) * (sB + 1)),
    "external": -1,
  }
  add_quality(nc, d)

  edge_quality = 2.0
  if nc["children"]:
    child_sizes = sum(child["size"] for child in nc["children"])
    if child_sizes > 0:
      edge_quality = nc["quality"] / (
        sum(child["quality"] * child["size"] for child in nc["children"])
      / child_sizes
      )

  NEXT_CLUSTER_ID += 1

  # add the edge with its quality annotated
  nc["edges"] |= { (fr, to, d, edge_quality) }

  return nc

def add_quality(cluster, new_edge_length):
  """
  Determine the quality of a cluster, according to the ratio between its
  internal-measure and a reference internal-measure (see the join_clusters
  function above), and add it to that cluster under the key "quality".

  For clusters whose internal-measure is zero (i.e., clusters without edges or
  clusters composed entirely of duplicates) we set the quality to 1.0.

  The reference measure is the internal-measure for cluster of equivalent size
  where the largest edge on the (shortest) path between any two points in the
  cluster is equal to the mean edge length within the cluster. We compare that
  against the actual internal-measure for the cluster, which is just the sum of
  the maximum-length edge on the (shortest) path between every pair of points
  in the cluster.
  """
  i = cluster["internal"]
  if i == 0: # The cluster contains no non-duplicates
    cluster["quality"] = 1
    return

  s = cluster["size"] + 1 # number of points rather than edges
  m = cluster["mean"]
  lengths = np.asarray(
    [d for (fr, to, d, q) in cluster["edges"]]
  + [new_edge_length]
  )
  std = np.std(lengths)
  tp = (s * (s - 1)) / 2 # total point pairs in the cluster
  ref = m * tp # the reference value described above
  softref = (m + std) * tp
  cluster["quality"] = ref / i
  cluster["soft_quality"] = softref / i

def add_edge(unions, clusters, best, edge, minsize, remember, memory_params):
  """
  Attempts to add the given edge to the given set of clusters (using the given
  unionfind data structure to detect cycle-inducing edges). Maintains a
  dictionary of the best clusters found so far, merging best clusters when they
  remain good and preserving them when a merge decreases quality significantly.

  Returns True if an edge was added and False if it was discarded due to the
  no-cycles constraint.
  """
  (fr, to, d) = edge
  r1 = unions.find(fr)
  r2 = unions.find(to)
  c1 = clusters[r1]
  c2 = clusters[r2]

  if r1 == r2: # if this edge would create a cycle; ignore it
    return False

  joined = join_clusters(c1, c2, (fr, to, d))

  # add the edge and combine the clusters it connects
  del clusters[r1]
  del clusters[r2]
  c1["external"] = d
  c2["external"] = d
  unions.unite(fr, to)
  nr = unions.find(fr)
  clusters[nr] = joined

  joined["children"] = []

  if c1["size"] >= minsize:
    c1["parent"] = joined
    joined["children"].append(c1)

  if c2["size"] >= minsize:
    c2["parent"] = joined
    joined["children"].append(c2)

  joined["dominance"] = (
    (1 + math.log(joined["size"]))
  / (1 + math.log(max(cl["size"] for cl in clusters.values())))
  )

  # Subcluster memory: don't retain all possible clusters in order to speed up
  # post-processing.
  if remember == "quality":
    threshold = memory_params["join_quality_threshold"]
    if c1["id"] in best:
      if (joined["quality"] / best[c1["id"]]["quality"]) > threshold:
        del best[c1["id"]]
        joined["children"].remove(c1)
        for child in c1["children"]:
          child["parent"] = joined
          joined["children"].append(child)
        c1["parent"] = None
        best[joined["id"]] = joined

    if c2["id"] in best:
      if (joined["quality"] / best[c2["id"]]["quality"]) > threshold:
        del best[c2["id"]]
        joined["children"].remove(c2)
        for child in c2["children"]:
          child["parent"] = joined
          joined["children"].append(child)
        c2["parent"] = None
        best[joined["id"]] = joined

  elif remember == "size":
    abssize = memory_params["absolute_size_threshold"]
    relsize = memory_params["relative_size_threshold"]
    if (
      c1["size"] < abssize
   or c2["size"] < abssize
   or c1["size"] < relsize * joined["size"]
   or c2["size"] < relsize * joined["size"]
    ): 
      if c1["id"] in best:
        del best[c1["id"]]
      if c2["id"] in best:
        del best[c2["id"]]
      if c1 in joined["children"]:
        joined["children"].remove(c1)
      if c2 in joined["children"]:
        joined["children"].remove(c2)
      for child in c1["children"]:
        child["parent"] = joined
        joined["children"].append(child)
      for child in c2["children"]:
        child["parent"] = joined
        joined["children"].append(child)
      c1["parent"] = None
      c2["parent"] = None

  elif remember == "size_and_scale":
    size_threshold = memory_params["absolute_size_threshold"]
    scale_threshold = memory_params["scale_increase_threshold"]
    incremental = (
      c1["size"] < size_threshold
   or c2["size"] < size_threshold
    )
    if d / c1["scale"] > scale_threshold:
      incremental = False
    if d / c2["scale"] > scale_threshold:
      incremental = False

    if incremental:
      if c1["id"] in best:
        del best[c1["id"]]
      if c2["id"] in best:
        del best[c2["id"]]
      if c1 in joined["children"]:
        joined["children"].remove(c1)
      if c2 in joined["children"]:
        joined["children"].remove(c2)
      for child in c1["children"]:
        child["parent"] = joined
        joined["children"].append(child)
      for child in c2["children"]:
        child["parent"] = joined
        joined["children"].append(child)
      c1["parent"] = None
      c2["parent"] = None

  # otherwise remember all subclusters (sill subject to some size filtering)

  if joined["size"] >= minsize:
    best[joined["id"]] = joined

  return True


def get_distances(points, metric="euclidean", normalize=0):
  """
  Computes a distance matrix for the given points containing all pairwise
  distances using the given metric. If normalize is given, it should be an
  integer indicating the neighborhood size for distance normalization.
  """
  distances = pairwise.pairwise_distances(points, metric=metric)

  if normalize:
    neighbors = NearestNeighbors(
      n_neighbors=normalize,
      algorithm="ball_tree"
    ).fit(points)

    norm_distances, norm_indices = neighbors.kneighbors(points)
    local_scale = np.median(norm_distances, axis=1)
    normalized = np.zeros_like(distances)

    for i in range(distances.shape[0]):
      for j in range(distances.shape[1]):
        normalized[i,j] = distances[i,j] / ((local_scale[i] + local_scale[j])/2)

    return normalized

  else:
    return distances

def get_neighbor_edges(nbd, nbi):
  """
  Computes all edges between nearest neighbors using the given
  nearest-neighbors matrices (see neighbors_from_points below).
  """
  return [
    (
      fr,
      nbi[fr][tidx],
      nbd[fr,tidx]
    )
      for fr in range(nbi.shape[0])
      for tidx in range(1,nbi.shape[1])
  ]

def neighbors_from_points(points, max_neighbors, metric="euclidean"):
  """
  Computes nearest neighbors (in the form of neighbor-distance and
  neighbor-index matrices) from points. Just a thin wrapper around
  sklearn.neighbors.NearestNeighbors.
  """
  neighbors = NearestNeighbors(
    n_neighbors=max_neighbors,
    metric=metric,
    algorithm="ball_tree",
  ).fit(points)

  return neighbors.kneighbors(points)

def neighbors_from_edges(n, edges, max_neighbors):
  """
  Takes n (the number of points) and an arbitrary edge list and produces
  a pair of neighbor-distance and neighbor-index arrays. Assumes that
  self-edges are not included in the edges list.
  """
  nbd = [[]] * n
  nbi = [[]] * n
  for x, (fr, to, d) in enumerate(edges):
    # TODO: DEBUG for this
    utils.prbar(x / len(edges))
    i = 0
    j = 0
    # forward
    while nbd[fr][i:] and nbd[fr][i] < d:
      i += 1
    nbd[fr].insert(i, d)
    nbi[fr].insert(i, to)
    nbd[fr] = nbd[fr][:max_neighbors]
    nbi[fr] = nbi[fr][:max_neighbors]

    # backward
    while nbd[to][j:] and nbd[to][j] < d:
      j += 1
    nbd[to].insert(j, d)
    nbi[to].insert(j, fr)
    nbd[to] = nbd[to][:max_neighbors]
    nbi[to] = nbi[to][:max_neighbors]

  # TODO: DEBUG for this
  print()

  nsize = max(len(l) for l in nbd)

  deficient = 0
  for i in range(len(nbd)):
    while len(nbd[i]) < nsize:
      deficient += 1
      nbd[i].append(np.inf)
      nbi[i].append(-1)

  if deficient:
    print(
      "Warning: given edges include {} deficiencies.",
      file=sys.stderr
    )

  return np.asarray(nbd), np.asarray(nbi)

@utils.default_params(DEFAULT_CLUSTERING_PARAMETERS)
def multiscale_clusters(points, **params):
  """
  This is the algorithm for computing multiscale clusters. It is a modified
  version of Kruskal's minimum-spanning-tree construction method which adds
  edges to a graph starting with the shortest edges and avoiding cycle-forming
  edges until a complete tree is formed.

  In this modified algorithm, edges which combine high-quality clusters into a
  low-quality cluster are skipped and revisited later, ultimately being thrown
  out if their quality-decreasing property persists. The result is a
  pseudo-minimum spanning forest of the data, which partitions it into
  clusters. Because only local distances are considered, clusters can have any
  shape.
  """
  global NEXT_CLUSTER_ID
  n = len(points)
  debug = utils.get_debug(params["quiet"])
  debug("Starting clustering process...")
  if params["edges"] is None:
    debug("  ...computing edges...")
    if params["distances"] is None:
      debug("  ...computing pairwise distances...")
      distances = get_distances(
        points,
        metric=params["metric"],
        normalize=(
          params["neighborhood_size"]
            if params["normalize_distances"]
            else 0
        )
      )
      debug("  ...done.")
    else:
      debug("  ...using given pairwise distances...")
      distances = params["distances"]

    debug("  ...building edge list...")
    edges = []
    for fr in range(n):
      for to in range(fr+1, n):
        edges.append((fr, to, distances[fr,to]))
        edges.append((to, fr, distances[to,fr]))

    edge_count = len(edges)
    debug("  ...found {} edges...".format(edge_count))
  else:
    edges = params["edges"]
    edge_count = len(edges)
    debug("  ...using {} given edges...".format(edge_count))

  debug(
    "  ...computing {} nearest neighbors from points...".format(
      params["neighborhood_size"]
    )
  )
  nbd, nbi = utils.cached_values(
    lambda: neighbors_from_points(
      points,
      params["neighborhood_size"],
      params["metric"]
    ),
    (
      params["neighbors_cache_name"] + "-distances",
      params["neighbors_cache_name"] + "-indices",
    ),
    ("pkl", "pkl"),
    override=not params["cache_neighbors"],
    debug=debug
  )
  debug(
    "  ...constructed lists for {} nearest neighbors...".format(nbd.shape[1])
  )

  debug("  ...sorting edges...")
  sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
  debug("  ...done.")

  u = uf.unionfind(n)

  debug("  ...creating point clusters...")
  clusters = {}
  for i in range(n):
    clusters[i] = {
      "id": i,
      "parent": None,
      "children": [],
      "size": 0,
      "scale": 0,
      "vertices": { i },
      "isolation": { i: (0, 1) },
      "outliers": set(),
      "core_size": 1,
      "edges": set(),
      "mean": 0,
      "internal": 0,
      "external": -1,
      "quality": 1
    }

  NEXT_CLUSTER_ID = n

  debug("  ...done.")
  debug("  ...constructing minimum spanning tree...")

  added = 0
  src = sorted_edges
  best = {}
  for e in src:
    if added == n - 1: # stop early
      break
    utils.prbar(added / (n - 1), debug=debug)

    # Add the next edge:
    added += int(
      add_edge(
        u,
        clusters,
        best,
        e,
        params["absolute_size_threshold"],
        params["remember"],
        {
          "absolute_size_threshold": params["absolute_size_threshold"],
          "relative_size_threshold": params["relative_size_threshold"],
          "join_quality_threshold": params["join_quality_threshold"],
          "scale_increase_threshold": params["scale_increase_threshold"],
        }
      )
    )

  debug()

  if added < n - 1:
    debug("  ...warning: graph was not completely connected...")

  debug("  ...done. Found {} clusters...".format(len(best)))
  debug("...done with clustering.")

  debug("Analyzing cluster statistics...")
  # Add normalized quality information:
  max_adj = 0
  max_coh = 0
  debug("  ...adding extra quality and coherence information...")
  for k in best:
    cl = best[k]
    cl["adjusted_quality"] = cl["quality"] * math.log(cl["core_size"])
    if max_adj < cl["adjusted_quality"]:
      max_adj = cl["adjusted_quality"]

    if cl["isolation_mean"] > 0:
      cl["coherence"] = cl["mean"] / cl["isolation_mean"]
    else:
      cl["coherence"] = 1 # mean must also be zero in this case

    if cl["external"] > 0:
      cl["obviousness"] = 1 - (cl["isolation_mean"] / cl["external"])
    elif cl["external"] == 0:
      cl["obviousness"] = 1.0
    else: # no external measure available: use scale as a proxy
      cl["obviousness"] = 1 - (cl["isolation_mean"] / cl["scale"])

    if max_coh < cl["coherence"]:
      max_coh = cl["coherence"]

  for k in best:
    cl = best[k]
    cl["norm_quality"] = cl["adjusted_quality"] / max_adj
    cl["norm_coherence"] = cl["coherence"] / max_coh

  debug("  ...done.")

  # Add compactness information:
  debug("  ...adding compactness and separation information...")
  for num, k in enumerate(best):
    utils.prbar(num / len(best), debug=debug)
    cl = best[k]

    # 1/2^n times number of nth-nearest-neighbors which are in the same cluster:
    cl["compactness"] = 0
    # Ratio between isolation (internal chain distance) and distance to nearest
    # out-of-cluster neighbor:
    cl["separation"] = 0
    nsep = 0

    for v in cl["vertices"]:
      outer_distance = -1

      for i, nb in enumerate(nbi[v]):
        if nb not in cl["vertices"]:
          cl["compactness"] += 1 / 2**i
          if outer_distance < 0:
            outer_distance = nbd[v][i]

      if outer_distance > 0: # found an outer neighbor
        cl["separation"] += cl["isolation"][v][0] / outer_distance
        nsep += 1

    n = cl["size"] + 1

    # Worst-case compactness is 1/2 + 1/4 + 1/8 + ... = 1 for each of n points
    # (Given that every nearest-neighbor is in the same cluster by construction)
    # Here we divide by this theoretical maximum and invert:
    cl["compactness"] = 1 - (cl["compactness"] / n)

    # Separation is nsep in the best-case, where the outer distance of each
    # edge vertex is equal to its isolation. If there are no edge vertices, we
    # count the cluster as perfectly-separated.
    if nsep > 0:
      cl["separation"] = 1 - (cl["separation"] / nsep)
    else:
      cl["separation"] = 1.0

  debug("  ...done.")

  for k in best:
    cl = best[k]

    cl["mixed_quality"] = (
    #  cl["quality"]
      cl["coherence"]
    * cl["dominance"]
    #* cl["obviousness"]
    #* cl["compactness"]
    #* cl["separation"]
    )

    cl["adjusted_mixed"] = (
      cl["mixed_quality"]
    * math.log(cl["size"])
    )

  debug("...done with analysis.")

  return best

def measure_deviation(
  clusters,
  stat="quality",
  against="size",
  fit="linear"
):
  """
  Adds a deviation measure to the given property by doing a regression of that
  property ordered by the given other property and measuring the difference
  between each point and the line. The regression is either linear (the
  default) or general exponential, according to the 'fit' parameter (which
  should be one of the strings "linear" or "exponential"). The added measure is
  named:

    <stat>_<against>_deviation
  """
  ordered = sorted(list(clusters.values()), key=lambda cl: cl[against])
  x = np.asarray([cl[against] for cl in ordered])
  y = np.asarray([cl[stat] for cl in ordered])

  if fit == "exponential":
    def expf(x, a, b, c):
      return a * np.exp(b*x) + c
    params, var = curve_fit(
      expf,
      x,
      y,
      p0=(0.5, -0.5, 0.5),
      bounds=((0, -1, 0), (1, 1, 1))
    )
    f = lambda x: expf(x, *params)
  elif fit == "linear":
    reg = linregress(x, y)
    f = lambda x: reg.intercept + reg.slope * x
  else:
    raise ValueError(
      (
        "Bad 'fit' parameter value '{}'"
        " (must be either 'linear' or 'exponential')."
      ).format(fit)
    )

  for k in clusters:
    cl = clusters[k]
    cl[stat + "_" + against + "_deviation"] = cl[stat] - f(cl[against])

def retain_best(clusters, filter_on="mixed_quality"):
  """
  Filters a group of clusters on a particular property (mixed_quality by
  default) by retaining only clusters which exceed the mean value.

  TODO: don't mess with parent/children links!
  """
  mean = np.mean([clusters[k][filter_on] for k in clusters])
  filtered = {
    k: clusters[k]
      for k in clusters if clusters[k][filter_on] >= mean
  }

  for k in filtered:
    cl = filtered[k]
    cl["children"] = []
    while cl["parent"] != None and cl["parent"]["id"] not in filtered:
      cl["parent"] = cl["parent"]["parent"]

  for k in filtered:
    if filtered[k]["parent"]:
      filtered[k]["parent"]["children"].append(filtered[k])

  return filtered

def retain_above(clusters, threshold=0.5, filter_on="mixed_quality"):
  """
  Filters a group of clusters on a particular property (mixed_quality by
  default) by retaining only clusters which exceed the given threshold.
  """
  filtered = {
    k: clusters[k]
      for k in clusters if clusters[k][filter_on] >= threshold
  }

  return filtered

def reassign_cluster_ids(clusters):
  newid = 0
  newgroup = {}
  for k in clusters:
    newgroup[newid] = clusters[k]
    clusters[k]["id"] = newid
    newid += 1

  return newgroup

def condense_best(clusters, threshold=0.9, quality="mixed_quality"):
  """
  Filters a group of clusters by for each parent, retaining descendants whose
  quality ratio is above the given threshold. Descendants with higher qualities
  increase the threshold but descendants with lower qualities don't lower it.
  """
  roots = find_all_largest(clusters)

  parents = [(r, r[quality]) for r in roots]
  next_parents = []
  results = roots

  considered = 0
  while parents:
    utils.prbar(considered / len(clusters))
    for cl, q in parents:
      for child in cl["children"]:
        considered += 1
        if child[quality] / q >= threshold:
          results.append(child)
          next_parents.append((child, child[quality]))
        else:
          next_parents.append((child, q))

    parents = next_parents
    next_parents = []

  return reassign_cluster_ids({ r["id"]: r for r in results })

def decant_best(clusters, quality="quality"):
  """
  Filters a group of clusters into a non-overlapping group, by picking the best
  clusters first according to cluster quality and then ignoring clusters that
  would cause overlap.
  """
  results = {}
  assigned = set()

  srt = reversed(sorted(list(clusters.values()), key=lambda cl: cl[quality]))

  newid = 0
  for cl in srt:
    if cl["vertices"] & assigned:
      continue
    assigned |= cl["vertices"]
    cl["id"] = newid
    results[newid] = cl
    newid += 1

  return results

def decant_split(clusters, threshold=1.0, criterion=lambda cl: 2.0):
  """
  Filters a group of clusters into a non-overlapping group, choosing to split
  up clusters according to the given criterion function and threshold (see some
  helpful examples below). If the criterion function returns a number higher
  than the threshold, the cluster is split.
  """
  results = {}
  assigned = set()

  srt = sorted(list(clusters.values()), key=lambda cl: -cl["size"])

  roots = []
  rooted = set()
  for cl in srt:
    if not cl["vertices"] & rooted:
      roots.append(cl)
      rooted |= cl["vertices"]

  clset = [ ]
  nextset = roots

  while len(nextset) > len(clset):
    clset = nextset
    nextset = []
    for cl in clset:
      if cl["size"] == 0 or len(cl["children"]) == 0: # can't split
        cl["split_quality"] = 0
        nextset.append(cl)
        continue
      cl["split_quality"] = criterion(cl)
      if cl["split_quality"] > threshold:
        nextset.extend(cl["children"])
      else:
        nextset.append(cl)

  # renumber the clusters:
  for i, cl in enumerate(clset):
    cl["id"] = i

  return { cl["id"]: cl for cl in clset }

def find_largest(clusters):
  """
  Returns the largest cluster from a group of clusters.
  """
  by_size = list(sorted(list(clusters.values()), key=lambda cl: cl["size"]))

  return by_size[-1]

def find_all_largest(clusters):
  """
  Returns a list of non-overlapping largest clusters from the given group.
  """

  srt = sorted(list(clusters.values()), key=lambda cl: -cl["size"])

  roots = []
  rooted = set()
  for cl in srt:
    if not cl["vertices"] & rooted:
      roots.append(cl)
      rooted |= cl["vertices"]

  return roots

def decant_erode(points, clusters, threshold=0.9):
  """
  Reduce a minimum spanning tree to a set of clusters by removing low-quality
  edges.
  """

  root = find_largest(clusters)

  retain = []
  for (fr, to, d, q) in root["edges"]:
    if q >= threshold:
      retain.append((fr, to, d))


  n = len(points)

  u = uf.unionfind(n)

  clusters = {}
  for i in range(n):
    clusters[i] = {
      "id": i,
      "parent": None,
      "children": [],
      "size": 0,
      "scale": 0,
      "vertices": { i },
      "isolation": { i: (0, 1) },
      "outliers": set(),
      "core_size": 1,
      "edges": set(),
      "mean": 0,
      "internal": 0,
      "external": -1,
      "quality": 1
    }

  best = {}
  added = 0
  for e in retain:
    # TODO: Debug control
    utils.prbar(added / len(retain))

    added += int(
      add_edge(
        u,
        clusters,
        best,
        e,
        0,
        0,
        None
      )
    )

  print() # done with progress bar

  return decant_best(clusters, quality="size")

def quality_vs_coverage_criterion(size="size", quality="quality"):
  """
  A splitting criterion for the decant_split function above. Compares the
  quality gained by splitting to the coverage (total size) lost. The given
  fields will be used as 'size' and 'quality' for the calculation. Pass the
  result of this function into decant_split.
  """
  def criterion(cluster):
    nonlocal size, quality
    child_coverage_ratio = (
      sum(child[size] for child in cluster["children"])
    / cluster[size]
    )
    child_quality_ratio = (
      max(child[quality] for child in cluster["children"])
    #  (
    #    sum(child[quality] for child in cluster["children"])
    #  / len(cluster["children"])
    #  )
    / cluster[quality]
    )
    return child_coverage_ratio * child_quality_ratio

  return criterion

def quality_vs_outliers_criterion(quality="quality"):
  """
  A splitting criterion for the decant_split function above. Compares the
  quality gained by splitting to the change in outlier percentage. The given
  fields will be used as the 'quality' for the calculation. Pass the result of
  this function into decant_split.
  """
  def criterion(cluster):
    nonlocal quality
    child_outliers_ratio = (
      (
        sum(child["size"] / child["core_size"] for child in cluster["children"])
      / len(cluster["children"])
      )
    / (cluster["size"] / cluster["core_size"])
    )
    child_quality_ratio = (
      max(child[quality] for child in cluster["children"])
    #  (
    #    sum(child[quality] for child in cluster["children"])
    #  / len(cluster["children"])
    #  )
    / cluster[quality]
    )
    return child_outliers_ratio * child_quality_ratio

  return criterion

def generational_criterion(feature="quality"):
  """
  A splitting criterion function that just uses the ratio of a single quantity
  between one generation and the next. If the given feature is a string, it's
  used to index clusters, otherwise it should be a function that gives a value
  for a cluster.
  """
  def criterion(cluster):
    nonlocal feature
    if type(feature) == str:
      return (
        (
          sum(child[feature] for child in cluster["children"])
        / len(cluster["children"])
        )
      / cluster[feature]
      )
    else:
      return (
        (
          sum(feature(child) for child in cluster["children"])
        / len(cluster["children"])
        )
      / feature(cluster)
      )

  return criterion

def product_criterion(*subcriteria):
  """
  Returns a criterion which combines multiple other criteria by multiplying
  them together.
  """
  def criterion(cluster):
    nonlocal subcriteria
    result = 1
    for sc in subcriteria:
      result *= sc(cluster)

    return result

  return criterion

def satisfaction_criterion(quality="quality"):
  """
  A splitting criterion function using size times quality.
  """
  def criterion(cluster):
    nonlocal quality
    satisfaction = cluster["core_size"] * cluster[quality]
    children_satisfaction = sum(
      child["core_size"] * child[quality] for child in cluster["children"]
    )
    return children_satisfaction / satisfaction

  return criterion

def scale_quality_criterion(quality="quality"):
  """
  A splitting criterion function using quality and scale gap.
  """
  def criterion(cluster):
    nonlocal quality
    child_quality_ratio = (
      (
        sum(child[quality] for child in cluster["children"])
      / len(cluster["children"])
      )
    / cluster[quality]
    )
    child_scale_ratio = (
      cluster["scale"]
    / (
        sum(child["scale"] for child in cluster["children"])
      / len(cluster["children"])
      )
    )
    return child_quality_ratio * child_scale_ratio

  return criterion

def vote_refine(
  all_points,
  clusters,
  leftovers_allowance=0.75,
  outlier_penalty=1.0,
  satisfaction_decay=0.8,
  quality="quality"
):
  """
  Refines a set of clusters by having every point vote for the best-quality
  cluster it's a member of and electing one cluster at a time until few
  unrepresented points (or no clusters) remain.
  """
  n = len(all_points)
  vote_weights = { p: 1.0 for p in range(n) }
  remaining_clusters = dict(clusters)
  elected = {}
  while (
    sum(vote_weights.values()) / n > leftovers_allowance
and remaining_clusters
  ):
    votes = {}
    for p in range(n):
      total_quality = 0
      my_votes = {}
      for k in remaining_clusters:
        cl = remaining_clusters[k]
        if p in cl["vertices"]:
          q = cl[quality]
          penalty = 1.0
          if p in cl["outliers"]:
            penalty = outlier_penalty
          my_votes[k] = q * penalty
          total_quality += q * penalty

      normalized = { k: my_votes[k]/total_quality for k in my_votes }
      for k in normalized:
        if k in votes:
          votes[k] += normalized[k] * vote_weights[p]
        else:
          votes[k] = normalized[k] * vote_weights[p]

    def quality_score(k):
      return votes[k] / clusters[k]["size"]

    best_cluster = None
    for k in votes:
      if best_cluster == None or quality_score(k) > quality_score(best_cluster):
        best_cluster = k

    print(
      "Elected {} (size {})".format(
        best_cluster,
        clusters[best_cluster]["size"]
      )
    )

    chosen = clusters[best_cluster]
    elected[best_cluster] = chosen
    for cp in chosen["vertices"]:
      vote_weights[cp] *= satisfaction_decay
    del remaining_clusters[best_cluster]

  return elected

@utils.default_params(DEFAULT_CLUSTERING_PARAMETERS)
def separated_multiscale(points, **params):
  """
  Calls multiscale_clusters with the given parameters and then filters the
  results a bit before returning them.
  """
  initial_clusters = multiscale_clusters(points, **params)
  #top = retain_best(initial_clusters, filter_on=params["quality"])
  #top = retain_best(top, filter_on=params["quality"])
  #sep = decant_best(top, quality=params["quality"])
  top = condense_best(
    initial_clusters,
    threshold=params["condensation_threshold"],
    quality=params["condensation_quality"]
  )
  top = retain_above(
    top,
    threshold=params["interesting_size"],
    filter_on="size"
  )
  sep = decant_best(top, quality=params["decanting_quality"])
  return sep

def cluster_assignments(points, clusters):
  clusters = reassign_cluster_ids(clusters)
  assignments = []
  for p in range(len(points)):
    assigned = False
    for k in clusters:
      if p in clusters[k]["vertices"]:
        assigned = True
        assignments.append(k)
        break
    if not assigned:
      assignments.append(-1)

  return np.asarray(assignments)

@utils.default_params(DEFAULT_TYPICALITY_PARAMETERS)
def typicality(points, **params):
  """
  For each point, grows a shortest-path tree until a certain fraction of the
  entire dataset is included. Measures the average longest-link-length of this
  tree during its growth, and normalizes these values to compute the typicality
  of each point.

  Parameters:

    name (default): description

    quiet (True):
      Whether to suppress debugging information.

    neighbors (None):
      A precomputed pair of neighbor distances and indices, as returned from
      the neighbors_from_points function above. Neighbors will be computed
      using that function directly from the points if not given.

    metric ("euclidean"): 
      The metric to use for distance calculations. Should be a valid
      scikit-learn metric (either a string (usually faster) or a function).

    significant_fraction (0.01): 
      How much of the data to explore to determine the typicality of each
      point. Larger fractions take longer but may help make finer distinctions
      between highly-typical points.

    min_neighborhood_size (15): 
      The minimum number of nearest neighbors to consider. By default, a number
      of neighbors equal to the significant_fraction times the number of points
      is used, but if this value is larger, it will be used instead. A hard
      upper limit of n/2 is also used, so that for *very* small data sets the
      neighborhood size won't be too large.

    parallelism (16):
      The number of parallel processes to use to compute typicality of separate
      points (the pool size for multiprocessing.Pool). Set to 1 to use only a
      single process.
  """
  debug = utils.get_debug(params["quiet"])

  n = len(points)

  neighborhood_size = int(min(
    max(
      n * params["significant_fraction"],
      params["min_neighborhood_size"]
    ),
    n/2
  ))

  debug("Starting typicality assessment...")
  debug("  ...computing edges...")
  if params["neighbors"] is None:
    debug("  ...computing nearest neighbors...")
    nbd, nbi = neighbors_from_points(
      points,
      neighborhood_size,
      params["metric"]
    )
    debug("  ...done.")
  else:
    debug("  ...using given neighbors...")
    nbd, nbi = params["neighbors"]
    if not params["suppress_warnings"] and nbd.shape[1] < min_neighborhood_size:
      print(
        (
          "Warning: The given information only includes {} neighbors,",
          " but the target neighborhood size is {}."
        ).format(nbd.shape[1], min_neighborhood_size),
        file=sys.stderr
      )

  debug(
    "  ...building {} neighborhoods of size {}...".format(
      n,
      neighborhood_size
    )
  )

  if params["parallelism"] > 1:
    debug("  ...running {} parallel processes...".format(params["parallelism"]))
    with multiprocessing.Pool(params["parallelism"]) as pool:
      def args(i):
        return (
          i,
          (nbd, nbi),
          neighborhood_size,
          params["suppress_warnings"]
        )
      results = np.asarray(
        pool.starmap(
          assess_typicality,
          [args(i) for i in range(n)],
          chunksize=500
        )
      )

      max_typ = np.max(results)
  else:
    max_typ = -1
    results = np.zeros((n,))

    for i in range(n):
      utils.prbar(i/n, debug=debug)

      results[i] = assess_typicality(
        i,
        (nbd, nbi),
        neighborhood_size,
        suppress_warnings = params["suppress_warnings"]
      )

      if max_typ < results[i]:
        max_typ = results[i]

  debug() # done with progress bar

  return results / max_typ

def assess_typicality(src, neighbors, target_size, suppress_warnings=False):
  """
  Builds a minimum-cost tree rooted at the given node, returning the average
  longest-edge distance during this process. The given neighbors should be a
  distances/indices pair as returned by e.g., neighbors_from_points.
  """
  typicality = 0
  current_max = 0
  size = 0
  covered = { src }

  nbd, nbi = neighbors

  local = [
    (nbd[src,:][i], nbi[src,:][i])
      for i in range(nbd.shape[1])
  ]

  while size < target_size and len(local) > 0:
    # Find the closest neighbor:
    bd, bn = local[0]

    if bn in covered: # if it's already covered, ignore that edge
      local = local[1:]
    else: # otherwise grow the tree & add edges
      if bd > current_max:
        current_max = bd

      local = merge_outgoing(
        local[1:],
        [
          (nbd[bn,:][i], nbi[bn,:][i])
            for i in range(nbd.shape[1])
        ]
      )

      # update state
      covered.add(bn)
      typicality += current_max
      size += 1

  if not suppress_warnings and size < target_size:
    print(
      (
        "Warning: typicality assessment failed to reach target size "
        "({} / {})"
      ).format(
        size,
        target_size
      ),
      file=sys.stderr
    )

  return typicality / size

def merge_outgoing(A, B):
  """
  Merges two sorted outgoing-edges lists into a sorted combined list.
  """
  i, j = 0, 0
  result = []
  while i < len(A) and j < len(B):
    if A[i][0] < B[j][0]:
      result.append(A[i])
      i += 1
    else:
      result.append(B[j])
      j += 1

  while i < len(A):
    result.append(A[i])
    i += 1

  while j < len(B):
    result.append(B[j])
    j += 1

  return result

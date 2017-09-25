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

import unionfind as uf

import numpy as np

from scipy.stats import linregress
from scipy.optimize import curve_fit

from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path

DEFAULT_CLUSTERING_PARAMETERS = {
  "quiet": True,
  # Distance parameters:
  "distances": None,
  "edges": None,
  "metric": "euclidean",
  "normalize_distances": False,
  "neighborhood_size": 10,
  "neighbors_cache_name": "multiscale-neighbors",
  "use_cached_neighbors": True,
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
  "sifting_threshold": 1.0,
  "sifting_quality": "mixed_quality",
  "decanting_quality": "quality",
  "condensation_discriminant": "mean",
  "condensation_quality": "adjusted_mixed",
  "condensation_distinction": 0.3,
  "condensation_tolerance": 0.8,
  #"condensation_distinction": 0.05,
  #"condensation_tolerance": 0.5,
  "add_compactness_and_separation": False,
}

DEFAULT_TYPICALITY_PARAMETERS = {
  "quiet": True,
  "suppress_warnings": False,
  "neighbors": None,
  "metric": "euclidean",
  "significant_fraction": 0.01,
  "min_neighborhood_size": 15,
  "parallelism": 16,
  #"parallelism": 1,
}

DEFAULT_ISOLATION_PARAMETERS = {
  "quiet": True,
  "metric": "euclidean",
  "normalize": True,
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
  nearest-neighbors matrices (see neighbors_from_distances below).
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
    n_neighbors=min(len(points), max_neighbors),
    metric=metric,
    algorithm="ball_tree",
  ).fit(points)

  return neighbors.kneighbors(points)

def neighbors_from_distances(distances, neighborhood_size):
  """
  Takes a pairwise distance matrix and produces nearest-neighbor index and
  distance metrics up to the given number of neighbors.
  """
  nbi = np.argsort(distances, axis=1)[:,1:neighborhood_size+1]
  nbd = np.zeros_like(nbi, dtype=float)
  for i, row in enumerate(nbi):
    nbd[i] = distances[i][row]

  return nbd, nbi

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
    distances = np.full((n, n), np.inf)
    for fr, to, d in edges:
      distances[fr][to] = d
    debug("  ...using {} given edges...".format(edge_count))

  debug(
    "  ...computing {} nearest neighbors from distances...".format(
      params["neighborhood_size"]
    )
  )
  nbd, nbi = utils.cached_values(
    lambda: neighbors_from_distances(
      distances,
      params["neighborhood_size"]
    ),
    (
      params["neighbors_cache_name"] + "-distances",
      params["neighbors_cache_name"] + "-indices",
    ),
    ("pkl", "pkl"),
    override=not params["use_cached_neighbors"],
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
    cl["adjusted_quality"] = cl["quality"] * (1 + math.log(cl["core_size"]))
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
  if params["add_compactness_and_separation"]:
    debug("  ...adding compactness and separation information...")
    for num, k in enumerate(best):
      utils.prbar(num / len(best), debug=debug)
      cl = best[k]

      # 1/2^n times number of nth-nearest-neighbors which are in same cluster:
      cl["compactness"] = 0
      # Ratio between isolation (internal chain distance) and distance to
      # nearest out-of-cluster neighbor:
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

      # Worst-case compactness is 1/2 + 1/4 + 1/8 + ... = 1 for each of n
      # points (Given that every nearest-neighbor is in the same cluster by
      # construction) Here we divide by this theoretical maximum and invert:
      cl["compactness"] = 1 - (cl["compactness"] / n)

      # Separation is nsep in the best-case, where the outer distance of each
      # edge vertex is equal to its isolation. If there are no edge vertices,
      # we count the cluster as perfectly-separated.
      if nsep > 0:
        cl["separation"] = 1 - (cl["separation"] / nsep)
      else:
        cl["separation"] = 1.0

    debug() # done with progress bar
    debug("  ...done.")

  for k in best:
    cl = best[k]

    cl["mixed_quality"] = (
    #  cl["quality"]
      cl["coherence"]
    * cl["dominance"]
    #* cl["obviousness"]
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

def reassign_cluster_ids(clusters, order_by="id"):
  """
  Re-assigns arbitrary contiguous IDs to a group of clusters. If order_by is
  given, IDs are assigned according to that property. This also reassigns
  parent/child links to avoid linking out of the given cluster set.
  """
  newid = 0
  newgroup = {}

  if order_by:
    organized = sorted(list(clusters.values()), key=lambda cl: cl[order_by])
  else:
    organized = clusters.values()

  for cl in organized:
    newgroup[newid] = cl
    cl["id"] = newid
    newid += 1

  ncl = list(newgroup.values())
  # First erase child information
  for cl in ncl:
    if "children" in cl:
      del cl["children"]

  # Now re-attach parents and reassign children based on that
  for cl in ncl:
    if "parent" in cl:
      p = cl["parent"]
      while p not in ncl and p != None:
        p = p["parent"]

      cl["parent"] = p
      if p != None:
        if "children" in p:
          p["children"].append(cl)
        else:
          p["children"] = [ cl ]

  return newgroup

def sift_children(clusters, threshold=1.0, quality="mixed_quality"):
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

def condense_best(
  clusters,
  discriminant="mean",
  quality="quality",
  distinction=0.3,
  tolerance=0.8
):
  """
  Filters a group of clusters by picking the best clusters first according to
  cluster quality and then judging clusters that would cause overlap according
  to their quality and size relative to existing clusters.
  """
  results = {}

  srt = reversed(sorted(list(clusters.values()), key=lambda cl: cl[quality]))

  newid = 0
  for cl in srt:
    overlapping = [
      res
        for res in results.values()
        if res["vertices"] & cl["vertices"]
    ]
    include = False
    if overlapping:
      diffs = [
        (
          # scale difference:
          (
            abs(cl[discriminant] - ov[discriminant])
          / max(cl[discriminant], ov[discriminant])
          ) if max(cl[discriminant], ov[discriminant]) > 0 else 0,
          # quality difference:
          cl[quality] / ov[quality]
        )
          for ov in overlapping
      ]
      if all(
        #sd > distinction
        sd > distinction and qd > tolerance
          for (sd, qd) in diffs
      ):
        include = True
    else:
      include = True

    if include:
      cl["id"] = newid
      results[newid] = cl
      newid += 1

  return results

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
  top = sift_children(
    initial_clusters,
    threshold=params["sifting_threshold"],
    quality=params["sifting_quality"]
  )
  top = retain_above(
    top,
    threshold=params["interesting_size"],
    filter_on="size"
  )
  sep = decant_best(top, quality=params["decanting_quality"])
  return sep

@utils.default_params(DEFAULT_CLUSTERING_PARAMETERS)
def condensed_multiscale(points, **params):
  """
  Calls multiscale_clusters with the given parameters and then filters the
  results a bit before returning them. Unlike separated_multiscale, doesn't
  always result in exclusive clusters, but because of that it can detect nested
  clusters.

  It returns a dictionary of all the condensed clusters retained, mapping
  cluster IDs to clusters where IDs are sorted by cluster size.
  """
  debug = utils.get_debug(params["quiet"])
  debug("Developing initial clusters...")
  initial_clusters = multiscale_clusters(points, **params)
  debug(
    "...found {} initial clusters; retaining large ones...".format(
      len(initial_clusters)
    )
  )
  big = retain_above(
    initial_clusters,
    threshold=params["interesting_size"],
    filter_on="size"
  )
  debug(
    "...retained {} large clusters; condensing best ones...".format(
      len(big)
    )
  )
  best = condense_best(
    big,
    discriminant=params["condensation_discriminant"],
    quality=params["condensation_quality"],
    distinction=params["condensation_distinction"],
    tolerance=params["condensation_tolerance"]
  )
  debug(
    "...condensed {} best clusters; done clustering.".format(
      len(best)
    )
  )
  return reassign_cluster_ids(best, order_by="size")

def cluster_assignments(n_points, clusters):
  by_size = list(
    sorted(
      list(clusters.items()),
      key=lambda kcl: kcl[1]["size"]
    )
  )
  assignments = []
  for p in range(n_points):
    assigned = False
    for k, cl in by_size:
      if p in cl["vertices"]:
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

    quiet (True):
      Whether to suppress debugging information.

    suppress_warnings (False):
      Whether or not to suppress warning messages.

    neighbors (None):
      A precomputed pair of neighbor distances and indices, as returned from
      the neighbors_from_distances function above. Neighbors will be computed
      using that function directly from the distances if not given.

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
    distances = get_distances(points, params["metric"])
    nbd, nbi = neighbors_from_distances(distances, neighborhood_size)
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
      results = np.array(
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
  distances/indices pair as returned by e.g., neighbors_from_distances.
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

@utils.default_params(DEFAULT_CLUSTERING_PARAMETERS)
def isolation(points, **params):
  """
  Runs clustering to compute isolation and returns just the isolation of each
  point, adjusted for incomplete clustering.
  """
  clusters = reassign_cluster_ids(
    multiscale_clusters(points, **params),
    order_by="size"
  )

  return derive_isolation(points, clusters)

def derive_isolation(points, clusters):
  """
  Given points and clusters, derives an isolation value for each point from the
  isolation information stored within each cluster.
  """
  n = len(points)

  by_size = list(
    reversed(
      sorted(
        list(clusters.values()),
        key=lambda cl: cl["size"]
      )
    )
  )

  longest_edge = max(cl["scale"] for cl in clusters.values())

  isolations = []
  miniso = np.inf
  maxiso = 0
  iso_seen = 0
  for p in range(len(points)):
    isolation = None
    for cl in by_size:
      if p in cl["vertices"]:
        avg, count = cl["isolation"][p]
        isolation = (avg * count + longest_edge * (n - count)) / n
        if isolation > maxiso and isolation < np.inf:
          maxiso = isolation
        if isolation < miniso:
          miniso = isolation
        iso_seen += 1
        break

    if isolation is None:
      # A true outlier, not found in any cluster at all
      isolations.append(np.inf)
    else:
      isolations.append(isolation)

  result = np.array(isolations)
  result = (result - miniso) / (maxiso - miniso)
  result[result == np.inf] = 2.0

  return result

def make_pickleable(clusters, strip_parents=False):
  """
  Strips child information from clusters so that they're pickleable (otherwise
  you get infinite recursion). There's still a lot of overhead if you pickle a
  set of clusters because each stores all of its parents separately, but if
  that's an issue you can use "strip_parents=True", with the drawback of not
  being able to recover ancestry information, with the drawback of not being
  able to recover ancestry information, with the drawback of not being able to
  recover ancestry information, with the drawback of not being able to recover
  ancestry information, with the drawback of not being able to recover ancestry
  information.

  Note that this function edits the given clusters, rather than creating a
  copy. It returns them as well for convenience.
  """
  for cl in clusters.values():
    del cl["children"]
    if strip_parents:
      del cl["parent"]

  return clusters

def restore_lineages(clusters):
  """
  Attempts to restore children information for a group of clusters that just
  has parent links (e.g., as produced by the make_pickleable function or after
  unpacking pickled results from that function).

  Note that this function edits the given clusters, rather than creating a
  copy. It returns them as well for convenience.
  """
  for cl in clusters.values():
    if "parent" in cl:
      p = cl["parent"]
      if p != None:
        if "children" in p:
          if cl in p["children"]:
            raise RuntimeWarning(
              "Restoring lineages to clusters which already have them!"
            )
          else:
            p["children"].append(cl)
        else:
          p["children"] = [ cl ]
    else:
      raise RuntimeWarning(
        (
          "Cannot restore lineages to clusters missing parents. "
          "(Did you set strip_parents=True in make_pickleable?)."
        )
      )

  return clusters

def multiobjective_argsort(criteria, flip=False, stop_early=None, exclude=None):
  """
  Takes an array where each column represents a different criterion and each
  row represents a single item with scores for each criterion. Ranks the items
  separately for each criterion, and then returns an index array indexing them
  by their multiobjective rank, breaking ties towards earlier columns.

  The multiobjective rank of an item is the first index i at which it is
  smaller than at least i items on *all* of the given criteria.

  If flip is True, the sense of all of the criteria is flipped.

  If stop_early is given, it should be an integer > 0 and after finding that
  many items, the results will be returned early.

  If exclude is given, it should be an array with shape (N, X) where N is the
  same as the first dimension of the criteria and X is arbitrary. Each entry in
  this array should be a valid criteria index; when an item is chosen all items
  from its row in the exclude array will be excluded from future consideration
  (if they've not already been chosen themselves). If you want to exclude
  different numbers of points upon inclusion of each point, let X be the length
  of the longest exclude list and pad your exclude matrix with -1's, as these
  will never be valid indices, so excluding them doesn't do anything.
  """

  orderings = np.argsort(criteria, axis=0)
  if flip:
    orderings = np.flip(orderings, axis=0)

  passed = np.zeros_like(criteria, dtype=bool)

  if not (exclude is None):
    skip = set()

  results = []
  for i in range(orderings.shape[0]):
    nextrow = orderings[i,:]
    changed = []
    for j, idx in enumerate(nextrow):
      passed[idx,j] = True
      changed.append(idx)

    for idx in changed:
      if not (exclude is None) and idx in skip:
        continue
      if passed[idx,:].all():
        results.append(idx)
        if stop_early and len(results) >= stop_early:
          break
        if not (exclude is None):
          skip.update(exclude[idx,:])

    if stop_early and len(results) >= stop_early:
      break

  return results


def find_exemplars(
  points,
  categorizations,
  metric="euclidean",
  skip_neighbors=0,
  desired_exemplars=16,
  neighborhood_size=1000,
  distances=None,
  neighbors=None
):
  """
  Takes an array of points and a same-length array with category-assignments
  for those points. For each category, identifies the n most-exemplary members
  of that category, as defined by their separation from members of other
  categories and proximity to members of the same category.

  "distances" and/or "neighbors" can be given directly if they're already
  available.

  The "skip_neighbors" argument specifies how many neighbors of a different
  category to skip while determining centrality and separation. Should usually
  be a small integer; the effect is subtle and a bit complicated, but
  presumably not too significant in most cases.

  Returns a mapping from categories to lists of exemplars, where each exemplar
  is given as a triple of (index, centrality, separation, foil_index).

  Centrality is the number of same-category items closer to a given item than
  the nearest different-category item, or the nth-nearest if skip_neighbors is
  greater than zero. Centrality is thus an integer between zero and
  neighborhood_size, and if centrality is equal to neighborhood_size,
  separation will be given as infinity.

  Separation is the distance to the different-category item that was used to
  determine centrality.

  Foil index is the index of the closest out-of-category neighbor.
  """
  n = len(points)

  if neighbors is None:
    if distances is None:
      distances = pairwise.pairwise_distances(points, metric=metric)
    nbd, nbi = neighbors_from_distances(distances, neighborhood_size)
  else:
    nbd, nbi = neighbors
    neighborhood_size = nbd.shape[1]

  all_categories = set(categorizations)

  all_results = {}
  for c in all_categories:
    centralities = np.zeros((n,), dtype=int)
    separations = np.zeros((n,), dtype=float)
    foils = np.zeros((n,), dtype=int)

    shadows = nbi[:,:]
    the_ones = np.arange(n)[categorizations == c]
    the_other = set(np.arange(n)[categorizations != c])

    for i in the_ones:
      centrality = None
      separation = None
      skip = skip_neighbors
      for j in range(neighborhood_size):
        if nbi[i,j] in the_other:
          skip -= 1
          shadows[i,j] = -1 # don't shade items of other categories
          if skip < 0:
            foils[i] = j
            centrality = j - skip_neighbors
            separation = nbd[i,j]
            for k in range(j + 1, neighborhood_size):
              shadows[i,j] = -1 # don't shade things outside central region
            break

      if separation is None:
        centrality = neighborhood_size
        separation = np.inf

      centralities[i] = centrality
      separations[i] = separation

    criteria = np.stack([centralities, separations], axis=-1)
    best = multiobjective_argsort(
      criteria,
      flip=True,
      stop_early=desired_exemplars,
      exclude=shadows
    )

    all_results[c] = [
      (i, centralities[i], separations[i], foils[i])
        for i in best
        if i in the_ones
    ]

  return all_results


def find_alt_exemplars(
  points,
  categorizations,
  metric="euclidean",
  desired_exemplars=16,
  neighborhood_size=500,
  distances=None,
  neighbors=None
):
  """
  An alternate algorithm for finding exemplars that just optimizes for the
  fraction of your neighborhood_size closes neighbors who share a category with
  you and the median distance to those neighbors.

  "distances" and/or "neighbors" can be given directly if they're already
  available.

  Returns a mapping from categories to lists of exemplars, where each exemplar
  is given as a triple of (index, purity, neighborhood_scale).

  Purity is the fraction of neighbors that share a category, and scale is the
  median neighbor distance among all neighbors regardless of category.
  """
  n = len(points)

  if neighbors is None:
    if distances is None:
      distances = pairwise.pairwise_distances(points, metric=metric)
    nbd, nbi = neighbors_from_distances(distances, neighborhood_size)
  else:
    nbd, nbi = neighbors
    neighborhood_size = nbd.shape[1]

  all_categories = set(categorizations)

  purities = np.zeros((n,), dtype=int)
  scales = np.median(nbd, axis=1)

  all_results = {}
  for c in all_categories:
    shadows = nbi[:,:neighborhood_size//2]
    the_other = set(np.arange(n)[categorizations != c])

    for i in range(n):
      ocount = sum(idx in the_other for idx in nbi[i,:])
      purities[i] = ocount / neighborhood_size

    criteria = np.stack([purities, scales], axis=-1)
    best = multiobjective_argsort(
      criteria,
      flip=True,
      stop_early=desired_exemplars,
      exclude=shadows
    )

    all_results[c] = [
      (i, purities[i], scales[i])
        for i in best
    ]

  return all_results


def find_simple_representatives(
  points,
  metric="euclidean",
  distances=None,
  density=0.01,
  inclusion_threshold=0.15
):
  """
  Takes an array of points and returns indices for a representative sample of
  those points, which is roughly "density" as large. Works by finding the
  median distance to the density × nth neighbor and then picking
  representatives starting from points with the smallest sum-of-neighbor
  distances but excluding any points within the found median distance from an
  already-chosen representative. Any representatives with less than
  inclusion_threshold × density × n points within their radius are ignored.

  "distances" can be given directly if they're already available.

  Returns a mapping from point indices of representatives to sets of point
  indices of the points they represent.
  """
  n = len(points)

  cn = int(n * density)

  incl = int(inclusion_threshold * cn)

  if distances is None:
    distances = pairwise.pairwise_distances(points, metric=metric)
  nbd, nbi = neighbors_from_distances(distances, cn)

  # exclusion distance for each representative
  critical_distance = np.median(nbd[:,-1])

  # ordering based on sum of neighbor distances (more central points go first)
  dsums = np.sum(nbd, axis=1)
  dorder = np.argsort(dsums)

  # sorted x-coordinates to speed up identification of excluded points
  ordering = np.argsort(points[:,0]) # sort on first axis
  x_ordered = points[ordering,0] # sorted x-coordinates
  omask = np.ones_like(ordering, dtype=bool)

  reps = {}
  skip = set()

  # function for returning set of points shaded by another
  def shaded_by(point):
    nonlocal points, distances, ordering, x_ordered, omask, skip
    x = points[point,0]
    left = np.searchsorted(x_ordered[omask], x - critical_distance, "left")
    right = np.searchsorted(x_ordered[omask], x + critical_distance, "right")

    shaded = set()
    masked = []

    origidx = np.arange(len(ordering))

    for oi, other in zip(origidx[omask], ordering[omask][left:right]):
      # TODO: This shouldn't be necessary
      if other in skip:
        print("ERROR: Had to skip manually.")
        continue

      d = distances[point,other]
      if d < critical_distance:
        shaded.add(other)
        masked.append(oi)

    return shaded, masked

  # Now just iterate through the points in order of their distance sums, adding
  # representatives and removing shaded points as appropriate.
  for p in dorder:
    if p in skip:
      continue
    shd, msk = shaded_by(p)
    if len(shd) >= incl:
      reps[p] = shd
      #print(sorted(list(shd))[:18])
      #print(sorted(list(ordering[omask][msk]))[:18])
      #print("<")
      skip.update(shd) # includes p itself
      # We'd like to do this, but advanced indexing creates a copy, not a view!
      omask[msk] = False
    # else do nothing: this point still counts for other point's shadow sizes,
    # and we don't need to skip it as we won't come back to it

  # TODO: return unrepresented points as well?
  return reps


def watersheds(points, nbd, nbi, criterion):
  """
  Accepts an array that lists a criterion value for each point as well as
  neighbor distance and index arrays. Finds and returns a mapping from the
  local minima of the given graph in terms of the given criterion to the sets
  of indices in the watershed of each local minimum.

  Note: if the criterion has plateaus, some items on a plateau will be
  considered their own local minima and some will join with neighboring members
  of the plateau, depending on point ordering. The easiest way to avoid this is
  to apply a blurring function to the criterion between neighboring points,
  with either a radius or number of iterations appropriate to plateau size in
  the raw input. Of course, this operation may obfuscate some local minima if
  the blur strength is too large and/or the hillsides are too steep.
  """
  n = criterion.shape[0]

  ns = nbi.shape[1]

  min_gap = np.min(nbd[nbd > 0])

  results = {}

  u = uf.unionfind(n)

  minima = []
  pmap = {}
  for i in range(n):
    c = criterion[i]
    parent = None
    maxslope = 0
    for j in range(ns):
      nd = nbd[i,j]
      ni = nbi[i,j]
      if ni == -1:
        break # no more real neighbors here
      slope = (c - criterion[ni]) / max(min_gap, nd)
      if slope > maxslope:
        maxslope = slope
        parent = ni
      elif slope == 0 and maxslope == 0:
        if ni in pmap or ni in minima:
          parent = ni

    if not (parent is None):
      u.unite(i, parent)
      pmap[i] = parent
    else:
      minima.append(i)

  pm = {}
  for m in minima:
    results[m] = set()
    if u.find(m) in pm:
      print(
        "Warning: rep {} has multiple minima (at least {} and {}).".format(
          u.find(m),
          pm[u.find(m)],
          m
        ),
        file=sys.stderr
      )
    pm[u.find(m)] = m

  for i in range(n):
    results[pm[u.find(i)]].add(i)

  return results


def minimum_spanning_tree(
  points,
  metric="euclidean",
  distances=None,
  debug=print
):
  """
  Computes a minimum spanning tree of the given points using Kruskal's
  algorithm. Returns the result as two variable-width matrices indicating
  neighbor distances and indices, where the ith row of the matrix corresponds
  to the ith point in the given list. The width of these matrices is the
  maximum arity among nodes in the spanning tree, and missing entries have
  index values of -1 and distance values of np.inf.
  """
  n = len(points)

  debug("Building minimum spanning tree...")
  if distances is None:
    debug("  ...computing pairwise distances...")
    distances = get_distances(points, metric=metric)
    debug("  ...done.")
  else:
    debug("  ...accepted precomputed distances...")

  debug("  ...building edge list...")
  edges = []
  for fr in range(n):
    for to in range(fr+1, n):
      edges.append((fr, to, distances[fr,to]))
      edges.append((to, fr, distances[to,fr]))

  edge_count = len(edges)
  debug("  ...found {} edges.".format(edge_count))

  debug("  ...sorting edges...")
  sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
  debug("  ...done.")

  u = uf.unionfind(n)

  arities = np.zeros((n,), dtype=int)
  neighbors = []
  ndists = []

  added = 0
  debug("  ...constructing tree...")
  for fr, to, d in sorted_edges:
    if added == n - 1: # stop early
      break

    utils.prbar(added / (n - 1), interval=50, debug=debug)

    fru = u.find(fr)
    tou = u.find(to)

    if fru != tou:
      added += 1
      u.unite(fr, to)

      # add a column to our neighbors & distances lists if necessary:
      if len(neighbors) < arities[fr] + 1 or len(neighbors) < arities[to] + 1:
        neighbors.append(np.full((n,), -1, dtype=int))
        ndists.append(np.full((n,), np.inf, dtype=float))

      # add information to our neighbors array
      neighbors[arities[fr]][fr] = to
      ndists[arities[fr]][fr] = d
      neighbors[arities[to]][to] = fr
      ndists[arities[to]][to] = d

      arities[fr] += 1
      arities[to] += 1
    # else do nothing and check the next edge
  debug("  ...done.")

  return np.stack(ndists, axis=-1), np.stack(neighbors, axis=-1)


def minimum_spanning_forest(
  points,
  metric="euclidean",
  distances=None,
  debug=print
):
  """
  Computes a minimum spanning forest of the given points using a modified
  Kruskal's algorithm where trees larger than size 1 aren't allowed to join.
  Returns the result as two variable-width matrices indicating neighbor
  distances and indices, where the ith row of the matrix corresponds to the ith
  point in the given list. The width of these matrices is the maximum arity
  among nodes in the spanning tree, and missing entries have index values of -1
  and distance values of np.inf.
  """
  n = len(points)

  debug("Building minimum spanning tree...")
  if distances is None:
    debug("  ...computing pairwise distances...")
    distances = get_distances(points, metric=metric)
    debug("  ...done.")
  else:
    debug("  ...accepted precomputed distances...")

  debug("  ...building edge list...")
  edges = []
  for fr in range(n):
    for to in range(fr+1, n):
      edges.append((fr, to, distances[fr,to]))
      edges.append((to, fr, distances[to,fr]))

  edge_count = len(edges)
  debug("  ...found {} edges.".format(edge_count))

  debug("  ...sorting edges...")
  sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
  debug("  ...done.")

  scales = np.median(distances[:,1:5], axis=1)

  u = uf.unionfind(n)

  arities = np.zeros((n,), dtype=int)
  neighbors = []
  ndists = []

  added = 0
  scales = {}
  sizes = {}
  debug("  ...constructing tree...")
  for fr, to, d in sorted_edges:
    if added == n - 1: # stop early
      break

    utils.prbar(added / (n - 1), interval=50, debug=debug)

    fru = u.find(fr)
    tou = u.find(to)

    if fru == tou:
      continue

    if fru in scales and tou in scales:
      local_scale = max(
        scales[fru] if fru in scales else 0,
        scales[tou] if tou in scales else 0
      )
      print("  dc", fr, to, d, local_scale)
      if d >= local_scale:
        continue

    print("AE", fr, to)

    # else add this edge and unite the clusters it joins
    added += 1
    u.unite(fr, to)
    nu = u.find(fr)
    csize = 0
    csize += sizes[fru] if fru in sizes else 1
    csize += sizes[tou] if tou in sizes else 1
    sizes[nu] = csize

    cscale = d
    cscale += (sizes[fru] - 1) * scales[fru] if fru in scales else 0
    cscale += (sizes[tou] - 1) * scales[tou] if tou in scales else 0
    scales[nu] = cscale / (csize - 1)

    # add a column to our neighbors & distances lists if necessary:
    if len(neighbors) < arities[fr] + 1 or len(neighbors) < arities[to] + 1:
      neighbors.append(np.full((n,), -1, dtype=int))
      ndists.append(np.full((n,), np.inf, dtype=float))

    # add information to our neighbors array
    neighbors[arities[fr]][fr] = to
    ndists[arities[fr]][fr] = d
    neighbors[arities[to]][to] = fr
    ndists[arities[to]][to] = d

    arities[fr] += 1
    arities[to] += 1

  debug("  ...done.")

  return np.stack(ndists, axis=-1), np.stack(neighbors, axis=-1)


def find_watershed_reps(
  points,
  metric="euclidean",
  distances=None,
  debug=print
):
  """
  Takes an array of points and returns indices for a representative sample of
  those points. Works by finding the points which are local minima in terms of
  nearest-neighbor-distance on a minimum spanning tree.

  "distances" can be given directly if they're already available.

  Returns a mapping from point indices of representatives to sets of point
  indices of the points they represent.
  """
  n = len(points)

  if distances is None:
    distances = pairwise.pairwise_distances(points, metric=metric)

  nbd, nbi = minimum_spanning_tree(points, distances=distances, debug=debug)

  #neighborhood_scale = np.mean(distances[:,1:2], axis=1)
  neighborhood_scale = nbd[:,0].flatten()

  return watersheds(points, nbd, nbi, neighborhood_scale)

def find_path(fr, to, nbd, nbi):
  """
  Finds a path from the given "fr"om node to the given "to" node, according to
  the graph specified by the neighbor distances and indices given. Returns None
  if there is no such path.

  Just implements simple breadth-first search.
  """
  arities = nbi.shape[1] - np.sum(nbi == -1, axis=1)
  queue = list(nbi[fr,:arities[fr]])
  parents = { n: fr for n in queue }
  found = False
  while queue:
    node = queue.pop(0)

    neighbors = nbi[node,:arities[node]]
    for n in neighbors:
      if n not in parents and n not in queue:
        parents[n] = node
        if n == to:
          found = True
          break
        queue.append(n)

    if found:
      break

  if found:
    path = [to]
    cur = to
    while path[-1] != fr:
      path.append(parents[cur])
      cur = parents[cur]

    return path

  else:
    return None

def find_new_rep(fr, to, nbd, nbi):
  """
  Takes a from node, a to node (as indices), a neighbors distance matrix and a
  neighbors index matrix and returns a new index indicating a suitable node
  that can represent both of the given nodes. Searches for a node that's
  roughly equidistant between them along a shortest path (assumes the given
  nbd/nbi arrays describe a minimum spanning tree so that each path is a
  shortest path).
  """
  path = find_path(fr, to, nbd, nbi)
  if len(path) <= 2: # degenerate case; arbitrarily pick 1st (or only)
    return path[0]

  pd = []
  for i in range(1, len(path)):
    di = -1
    for j in range(nbi.shape[1]):
      if nbi[path[i-1],j] == path[i]:
        di = j
        break

    if di == -1: # shouldn't be possible
      print("Error finding new rep: invalid path!", file=sys.stderr)

    pd.append(nbd[path[i-1],di])

  # cumulative distances to the from & to nodes
  cdf = [sum(pd[:i]) for i in range(1,len(pd))]
  cdt = [sum(pd[i+1:]) for i in range(len(pd)-1)]

  cdd = [abs(cdf[i] - cdt[i]) for i in range(len(cdf))]

  best = 1 + np.argmin(cdd)

  return path[best]


def find_representatives(
  points,
  metric="euclidean",
  distances=None,
  debug=print
):
  """
  Takes an array of points and returns indices for a representative sample of
  those points.

  Starts by finding watershed representatives on a minimum spanning tree and
  proceeds to group these representatives together by joining nearby
  representatives and eroding their joint watersheds.

  "distances" can be given directly if they're already available.

  Returns a mapping from point indices of representatives to sets of point
  indices of the points they represent.
  """
  n = len(points)

  debug("Finding representatives...")

  if distances is None:
    debug("  ...computing pairwise distances...")
    distances = get_distances(points, metric=metric)
    debug("  ...done computing distances.")
  else:
    debug("  ...using given distances...")

  debug("  ...constructing minimum spanning tree...")
  nbd, nbi = minimum_spanning_tree(points, distances=distances, debug=debug)
  debug("  ...done constructing spanning tree.")

  debug("  ...finding candidate representatives...")
  neighborhood_scale = nbd[:,0].flatten()

  candidates = watersheds(points, nbd, nbi, neighborhood_scale)
  debug("  ...done finding candidates.")

  if len(candidates) < 2:
    debug("...found a singular representative.")
    return candidates

  med_cd = None

  # TODO: do this until convergence?
  debug("  ...merging representatives...")
  for i in range(3):
    imap = np.array(list(candidates.keys()))
    cpoints = points[imap]
    cnd, cni = neighbors_from_points(cpoints, 2, metric=metric)
    cnd = cnd[:,1]
    cni = cni[:,1]

    if med_cd == None: # only do this once
      #med_cd = np.percentile(cds, 25, interpolation="linear")
      med_cd = np.median(cnd)

    targets = cnd < med_cd
    debug("  ...found {} targets...".format(sum(targets)))
    merge = np.array(
      np.stack(
        [
          imap[targets],
          imap[cni[targets]],
          cnd[targets]
        ],
        axis=-1
      )
    )

    order = np.argsort(merge[:,2])

    debug("  ...merge targets are:")
    debug(merge[order])

    results = {}
    for (fr, to, d) in merge[order]:
      fr = int(fr)
      to = int(to)
      if fr not in candidates or to not in candidates:
        debug("    ...skipping preempted merge ({} and {})...".format(fr, to))
        continue

      debug("    ...merging reps {} and {} (d={})...".format(fr, to, d))
      nrep = find_new_rep(fr, to, nbd, nbi)
      results[nrep] = candidates[fr] | candidates[to]
      del candidates[fr]
      del candidates[to]

    for unmerged in candidates:
      results[unmerged] = candidates[unmerged]

    # swap and iterate again
    candidates = results

  debug("  ...done merging representatives.")
  debug("...done finding representatives.")

  return candidates
